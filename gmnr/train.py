"""Train pi-GAN. Supports distributed training."""


import copy
import json
import os
import random
import time
from datetime import datetime
import math
import joblib
import numpy as np
import torch

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_ema import ExponentialMovingAverage
from torch_fidelity import calculate_metrics
from torchvision.utils import save_image
from tqdm import tqdm

import gmnr.curriculums as curriculums
import gmnr.datasets as datasets
from gmnr.core.mpi_renderer import MPIRenderer
from gmnr.train_helpers import STYLEGAN2_CFG_SPECS, cleanup, find_worst_view_per_z, modify_curriculums, setup, z_sampler
from gmnr.utils import TensorboardWriter, convert_cfg_to_dict, logger
import torch.nn.functional as F
from gmnr.models.loses import *
import os



def train(rank, world_size, config, master_port, run_dataset, EXP_NAME):

    xyz_coords_only_z = False

    # https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113/2
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()

    # reproducibility set up
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch_g = torch.Generator()
    torch_g.manual_seed(config.SEED)

    modify_curriculums(config, run_dataset=run_dataset)

    # save config for reproducibility
    torch.save(
        {"config": convert_cfg_to_dict(config)},
        os.path.join(config.CHECKPOINT_FOLDER, "config.pth"),
    )

    if rank == 0:
        logger.add_filehandler(config.LOG_FILE)
        

    opt = config.gmnr.TRAIN
    opt.defrost()
    opt.port = master_port
    opt.freeze()

    setup(rank, world_size, opt.port)
    device = torch.device(rank)

    curriculum = getattr(curriculums, opt.curriculum)
    metadata = curriculums.extract_metadata(curriculum, 0)

    if rank == 0:
        logger.info(f"config: {config}\n\n")
        logger.info(f"curriculum: {curriculum}\n")

        writer = TensorboardWriter(config.TENSORBOARD_DIR, flush_secs=10)

    fixed_z = z_sampler((25, metadata["latent_dim"]), device="cpu", dist=metadata["z_dist"])

    if opt.load_dir != "":
        raise NotImplementedError
    else:

        n_g_out_channels = 4
        n_g_out_planes = config.gmnr.MPI.n_gen_planes

        # fmt: off
        if config.gmnr.MODEL.STYLEGAN2.torgba_cond_on_pos_enc != "none":
            if config.gmnr.MODEL.STYLEGAN2.torgba_cond_on_pos_enc == "depth2alpha":
                print("\nGenerator comes from depth2alpha\n")
                from gmnr.models.networks.networks_vanilla_depth2alpha import Generator as StyleGAN2Generator
            elif config.gmnr.MODEL.STYLEGAN2.torgba_cond_on_pos_enc == "normalize_add_z":
                if config.gmnr.MODEL.STYLEGAN2.torgba_cond_on_pos_enc_embed_func in ["learnable_param"]:
                    print("\nGenerator comes from learnable_param\n")
                    from gmnr.models.networks.networks_pos_enc_learnable_param import Generator as StyleGAN2Generator
                elif config.gmnr.MODEL.STYLEGAN2.torgba_cond_on_pos_enc_embed_func in ["modulated_lrelu"]:
                    print("\nGenerator comes from cond_on_depth\n")
                    from gmnr.models.networks.networks_cond_on_pos_enc import Generator as StyleGAN2Generator
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            print("\nGenerator comes from vanilla\n")
            from gmnr.models.networks.networks_vanilla import Generator as StyleGAN2Generator
        # fmt: on

        synthesis_kwargs = convert_cfg_to_dict(config.gmnr.MODEL.STYLEGAN2.synthesis_kwargs)
        synthesis_kwargs_D = convert_cfg_to_dict(config.gmnr.MODEL.STYLEGAN2.synthesis_kwargs)
        # NOTE: ref: https://github.com/NVlabs/stylegan2-ada-pytorch/blob/6f160b3d22b8b178ebe533a50d4d5e63aedba21d/train.py#L178
        synthesis_kwargs["channel_base"] = int(
            STYLEGAN2_CFG_SPECS[str(config.gmnr.MODEL.STYLEGAN2.max_out_dim)]["fmaps"]
            * synthesis_kwargs["channel_base"]
        )
        synthesis_kwargs_D["channel_base"] = int(
            STYLEGAN2_CFG_SPECS[str(config.gmnr.MODEL.STYLEGAN2.max_out_dim_D)]["fmaps"]
            * synthesis_kwargs_D["channel_base"]
        )

        # For clamping, ref:
        # - Sec. D.1 Mixed-precision training of https://arxiv.org/abs/2006.06676
        # - https://github.com/NVlabs/stylegan2-ada-pytorch/blob/6f160b3d22b8b178ebe533a50d4d5e63aedba21d/train.py#L333
        if config.gmnr.MODEL.STYLEGAN2.max_out_dim <= 128:
            synthesis_kwargs["num_fp16_res"] = 0
            synthesis_kwargs["conv_clamp"] = None

            synthesis_kwargs_D["num_fp16_res"] = 0
            synthesis_kwargs_D["conv_clamp"] = None
        else:
            synthesis_kwargs["conv_clamp"] = 256
            # Ref: https://github.com/NVlabs/stylegan2-ada-pytorch/blob/d4b2afe9c27e3c305b721bc886d2cb5229458eba/train.py#L181
            synthesis_kwargs["num_fp16_res"] = 4

            synthesis_kwargs_D["conv_clamp"] = 256
            synthesis_kwargs_D["num_fp16_res"] = 4

        if config.gmnr.TRAIN.normalized_xyz_range == "01":
            depth2alpha_z_range = 1.0
        elif config.gmnr.TRAIN.normalized_xyz_range == "-11":
            depth2alpha_z_range = 2.0
        else:
            raise ValueError

        generator = StyleGAN2Generator(
            z_dim=metadata["latent_dim"],  # config.gmnr.MODEL.STYLEGAN2.z_dim,
            c_dim=metadata["generator_label_dim"],  # config.gmnr.MODEL.STYLEGAN2.label_dim,
            w_dim=metadata["stylegan2_w_dim"],  # config.gmnr.MODEL.STYLEGAN2.w_dim,
            img_resolution=config.gmnr.MODEL.STYLEGAN2.max_out_dim,
            n_planes=n_g_out_planes,
            plane_channels=n_g_out_channels,
            pos_enc_multires=config.gmnr.MODEL.STYLEGAN2.pos_enc_multires,
            torgba_cond_on_pos_enc=config.gmnr.MODEL.STYLEGAN2.torgba_cond_on_pos_enc,
            torgba_cond_on_pos_enc_embed_func=config.gmnr.MODEL.STYLEGAN2.torgba_cond_on_pos_enc_embed_func,
            torgba_sep_background=config.gmnr.MODEL.STYLEGAN2.torgba_sep_background,
            build_background_from_rgb=config.gmnr.MODEL.STYLEGAN2.build_background_from_rgb,
            build_background_from_rgb_ratio=config.gmnr.MODEL.STYLEGAN2.build_background_from_rgb_ratio,
            cond_on_pos_enc_only_alpha=config.gmnr.MODEL.STYLEGAN2.cond_on_pos_enc_only_alpha,
            gen_alpha_largest_res=config.gmnr.MODEL.STYLEGAN2.gen_alpha_largest_res,
            background_alpha_full=config.gmnr.MODEL.STYLEGAN2.background_alpha_full,
            G_final_img_act=config.gmnr.MODEL.STYLEGAN2.G_final_img_act,
            depth2alpha_z_range=depth2alpha_z_range,
            depth2alpha_n_z_bins=config.gmnr.MPI.depth2alpha_n_z_bins,
            mapping_kwargs=convert_cfg_to_dict(config.gmnr.MODEL.STYLEGAN2.mapping_kwargs),
            synthesis_kwargs=synthesis_kwargs,
        ).to(device)

        if config.gmnr.TRAIN.D_from_stylegan2:
            assert not config.gmnr.TRAIN.D_pred_pos
            from gmnr.models.networks.networks_cond_on_pos_enc import Discriminator as DiscriminatorStyleGAN2

            if config.gmnr.TRAIN.D_cond_on_pose:
                stylegan2_cdim = config.gmnr.TRAIN.D_cond_pose_dim
            else:
                stylegan2_cdim = 0
            img_channels = 3
            discriminator = DiscriminatorStyleGAN2(
                stylegan2_cdim,
                config.gmnr.MODEL.STYLEGAN2.max_out_dim_D,
                img_channels,
                channel_base=synthesis_kwargs_D["channel_base"],
                channel_max=config.gmnr.MODEL.STYLEGAN2.synthesis_kwargs.channel_max,
                num_fp16_res=synthesis_kwargs_D["num_fp16_res"],
                conv_clamp=synthesis_kwargs_D["conv_clamp"],
                cmap_dim=config.gmnr.MODEL.STYLEGAN2.discriminator.cmap_dim,
                D_stylegan2_ori_mapping=config.gmnr.MODEL.STYLEGAN2.discriminator.use_ori_mapping,
                use_mbstd_in_D=config.gmnr.MODEL.STYLEGAN2.discriminator.use_mbstd_in_D,
                epilogue_kwargs={
                    "mbstd_group_size": config.gmnr.MODEL.STYLEGAN2.discriminator.mbstd_group_size,
                    # "mbstd_num_channels": config.gmnr.MODEL.STYLEGAN2.discriminator.mbstd_num_channels,
                },
            ).to(device)
        else:
            raise NotImplementedError

        ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
        ema2 = ExponentialMovingAverage(generator.parameters(), decay=0.9999)

    if config.gmnr.MODEL.use_pretrained_ckpt:
        import gmnr.models.legacy as stylegan2_legacy
        import gmnr.models.torch_utils.misc as stylegan2_misc
        from gmnr.models.dnnlib.util import open_url

        print(f'[rank {rank}] Resuming from "{config.gmnr.MODEL.pretrained}"')

        # NOTE: it requires torch_utils folder in PYTHONPATH
        with open_url(config.gmnr.MODEL.pretrained) as f:
            resume_data = stylegan2_legacy.load_network_pkl(f)

        print(f'[rank {rank}] Resuming D from "{config.gmnr.MODEL.pretrained_D}"')
        with open_url(config.gmnr.MODEL.pretrained_D) as f:
            resume_data_D = stylegan2_legacy.load_network_pkl(f)

        generator = generator.train().requires_grad_(False)
        discriminator = discriminator.train().requires_grad_(False)
        ema_placeholder = copy.deepcopy(generator)

        # NOTE: we resume G from G_ema
        # for name, module in [('G_ema', generator), ('G_ema', ema_placeholder)]:
        for name, module in [("G_ema", generator), ("G_ema", ema_placeholder)]:
            logger.info(f"\n\n[rank {rank}] Resume {name}\n")
            stylegan2_misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

        if config.gmnr.TRAIN.D_from_stylegan2:
            logger.info(f"\n\n[rank {rank}] Resume D\n")
            stylegan2_misc.copy_params_and_buffers(resume_data_D["D"], discriminator, require_all=False)

        generator = generator.train().requires_grad_(True)
        discriminator = discriminator.train().requires_grad_(True)

        ema = ExponentialMovingAverage(ema_placeholder.parameters(), decay=0.999)
        ema2 = ExponentialMovingAverage(ema_placeholder.parameters(), decay=0.9999)

    ssim =  SSIM().to(device)

    generator_ddp = DDP(
        generator,
        device_ids=[rank],
        find_unused_parameters=True,
        broadcast_buffers=False,
    )
    discriminator_ddp = DDP(
        discriminator,
        device_ids=[rank],
        find_unused_parameters=True,
        broadcast_buffers=False,
    )
    generator = generator_ddp.module
    discriminator = discriminator_ddp.module

    # NOTE: setup tuning strategy here
    if config.gmnr.TRAIN.only_tune:
        raise NotImplementedError
    else:
        assert config.gmnr.TRAIN.D_train
        enable_mapping_grad = True
        enable_syn_feat_net_grad = True

    assert not metadata.get("unique_lr", False)

    optimizer_G = torch.optim.Adam(
        generator_ddp.parameters(),
        lr=metadata["gen_lr"],
        betas=metadata["betas"],
        weight_decay=metadata["weight_decay"],
    )

    optimizer_D = torch.optim.Adam(
        discriminator_ddp.parameters(),
        lr=metadata["disc_lr"],
        betas=metadata["betas"],
        weight_decay=metadata["weight_decay"],
    )

    generator_losses = []
    discriminator_losses = []

    generator.set_device(device)

    # set up MPI renderer
    mpi_renderer = MPIRenderer(
        n_mpi_planes=config.gmnr.MPI.n_gen_planes,
        plane_min_d=metadata["ray_start"],
        plane_max_d=metadata["ray_end"],
        plan_spatial_enlarge_factor=config.gmnr.MPI.CAM_SETUP.spatial_enlarge_factor,
        plane_distances_sample_method=config.gmnr.MPI.distance_sample_method,
        cam_fov=metadata["fov"],
        sphere_center_z=config.gmnr.MPI.CAM_SETUP.cam_sphere_center_z,
        sphere_r=config.gmnr.MPI.CAM_SETUP.cam_sphere_r,
        horizontal_mean=metadata["h_mean"],
        horizontal_std=metadata["h_stddev"],
        vertical_mean=metadata["v_mean"],
        vertical_std=metadata["v_stddev"],
        cam_pose_n_truncated_stds=config.gmnr.MPI.CAM_SETUP.cam_pose_n_truncated_stds,
        cam_sample_method=config.gmnr.MPI.CAM_SETUP.cam_pose_sample_method,
        mpi_align_corners=config.gmnr.MPI.align_corners,
        use_xyz_ztype=config.gmnr.TRAIN.use_xyz_ztype,
        use_normalized_xyz=config.gmnr.TRAIN.use_normalized_xyz,
        normalized_xyz_range=config.gmnr.TRAIN.normalized_xyz_range,
        use_confined_volume=config.gmnr.MPI.use_confined_volume,
        device=device,
    )


    mpi_return_single_res_xyz = False

    # ----------
    #  Training
    # ----------

    if rank == 0:
        with open(os.path.join(opt.output_dir, "options.txt"), "w") as f:
            f.write(str(opt))
            f.write("\n\n")
            f.write(str(generator))
            f.write("\n\n")
            f.write(str(discriminator))
            f.write("\n\n")
            f.write(str(curriculum))

        # log information about the model
        import gmnr.models.torch_utils.misc as stylegan2_misc

        z = torch.empty([1, generator.z_dim], device=device)
        c = torch.empty([1, generator.c_dim], device=device)
        mpi_tex_pix_xyz, mpi_tex_pix_normalized_xyz = mpi_renderer.get_xyz(
            metadata["tex_size"], metadata["tex_size"], ret_single_res=mpi_return_single_res_xyz
        )
        if config.gmnr.TRAIN.use_normalized_xyz:
            mpi_xyz_input = mpi_tex_pix_normalized_xyz
        else:
            mpi_xyz_input = mpi_tex_pix_xyz

        torch.cuda.empty_cache()

        n_g_params = sum(p.numel() for p in generator.parameters())
        logger.info(f"#params of G: {n_g_params}\n")

        if config.gmnr.TRAIN.D_from_stylegan2:
            c = torch.empty([1, discriminator.c_dim], device=device)
            img_placeholder = torch.empty([1, 3, metadata["img_size"], metadata["img_size"]], device=device)
            alpha_placeholder = 1.0
            stylegan2_misc.print_module_summary(discriminator, [img_placeholder, alpha_placeholder, c])
        else:
            logger.info(f"discriminator: {discriminator}")

        n_d_params = sum(p.numel() for p in discriminator.parameters())
        logger.info(f"#params of D: {n_d_params}\n")

    torch.manual_seed(rank)
    dataloader = None
    total_progress_bar = tqdm(total=opt.total_iters, desc="Total progress", dynamic_ncols=True)
    total_progress_bar.update(discriminator.step)

    for epoch_i in range(opt.n_epochs):

        if discriminator.step > opt.total_iters:
            break

        metadata = curriculums.extract_metadata(curriculum, discriminator.step)

        # NOTE: for condition on pose
        metadata["sphere_center"] = config.gmnr.MPI.CAM_SETUP.cam_sphere_center_z
        metadata["sphere_r"] = config.gmnr.MPI.CAM_SETUP.cam_sphere_r
        metadata["flat_pose_dim"] = config.gmnr.TRAIN.D_cond_pose_dim

        # we reset camera if needed
        if metadata["img_size"] != mpi_renderer.render_h or metadata["img_size"] != mpi_renderer.render_w:
            mpi_renderer.set_cam(metadata["fov"], metadata["img_size"], metadata["img_size"])

        # Set learning rates
        for param_group in optimizer_G.param_groups:
            if param_group.get("name", None) == "mapping_network":
                param_group["lr"] = metadata["gen_lr"] * 5e-2
            else:
                param_group["lr"] = metadata["gen_lr"]
            param_group["betas"] = metadata["betas"]
            param_group["weight_decay"] = metadata["weight_decay"]
        for param_group in optimizer_D.param_groups:
            param_group["lr"] = metadata["disc_lr"]
            param_group["betas"] = metadata["betas"]
            param_group["weight_decay"] = metadata["weight_decay"]

        if not dataloader or dataloader.dataset.img_size != metadata["img_size"]:
            del dataloader
            time.sleep(10)

            dataloader, data_sampler, CHANNELS = datasets.get_dataset_distributed(
                metadata["dataset"],
                world_size,
                rank,
                num_workres=config.gmnr.TRAIN.n_dataloader_workers,
                torch_g=torch_g,
                **metadata,
            )

            step_next_upsample = curriculums.next_upsample_step(curriculum, discriminator.step)
            step_last_upsample = curriculums.last_upsample_step(curriculum, discriminator.step)

        logger.info(f"New epoch {epoch_i}.\n\n")

        # NOTE: this is requred to make distributed sampler shuffle dataset.
        data_sampler.set_epoch(epoch_i)

        for batch_i, batch_data in enumerate(dataloader):

            if discriminator.step > opt.total_iters:
                break

            if rank == 0:
                total_progress_bar.update(1)

            # pred_yaws_real/pred_pitches_real: [B, 1]
            imgs, flat_w2c_mats_real, _, pred_yaws_real, pred_pitches_real = batch_data
            # NOTE: we only condition on rotation
            flat_w2c_mats_real = flat_w2c_mats_real.to(device)
            pred_pose_angles_real = torch.cat((pred_pitches_real, pred_yaws_real), dim=1).to(device)

            cur_h_stddev = metadata["h_stddev"]
            cur_v_stddev = metadata["v_stddev"]

            if discriminator.step % opt.model_save_interval == 0 and rank == 0:
                # fmt: off
                now = datetime.now()
                now = now.strftime("%d--%H:%M--")
                torch.save(ema.state_dict(), os.path.join(config.CHECKPOINT_FOLDER, f'ema_{discriminator.step}.pth'))
                torch.save(ema2.state_dict(), os.path.join(config.CHECKPOINT_FOLDER, f'ema2_{discriminator.step}.pth'))
                torch.save(generator_ddp.module.state_dict(), os.path.join(config.CHECKPOINT_FOLDER, f'generator_{discriminator.step}.pth'))
                torch.save(discriminator_ddp.module.state_dict(), os.path.join(config.CHECKPOINT_FOLDER, f'discriminator_{discriminator.step}.pth'))
                torch.save(optimizer_G.state_dict(), os.path.join(config.CHECKPOINT_FOLDER, f'optimizer_G_{discriminator.step}.pth'))
                torch.save(optimizer_D.state_dict(), os.path.join(config.CHECKPOINT_FOLDER, f'optimizer_D_{discriminator.step}.pth'))
                # fmt: on

            metadata = curriculums.extract_metadata(curriculum, discriminator.step)

            # NOTE: for condition on pose
            metadata["sphere_center"] = config.gmnr.MPI.CAM_SETUP.cam_sphere_center_z
            metadata["sphere_r"] = config.gmnr.MPI.CAM_SETUP.cam_sphere_r
            metadata["flat_pose_dim"] = config.gmnr.TRAIN.D_cond_pose_dim

            # if dataloader.batch_size != metadata["batch_size"]:
            if dataloader.dataset.img_size != metadata["img_size"]:
                time.sleep(10)
                break

            generator_ddp.train()
            discriminator_ddp.train()

            if config.gmnr.TRAIN.D_from_stylegan2:
                alpha = 1.0
            else:
                alpha = min(1, (discriminator.step - step_last_upsample) / (metadata["fade_steps"]))
                # alpha = 1.0

            real_imgs = imgs.to(device, non_blocking=True)

            # TRAIN DISCRIMINATOR

            # if discriminator.step % 2 == 0:
            optimizer_D.zero_grad()

            with torch.set_grad_enabled(config.gmnr.TRAIN.D_train):

                mpi_tex_pix_xyz, mpi_tex_pix_normalized_xyz = mpi_renderer.get_xyz(
                    metadata["tex_size"], metadata["tex_size"], ret_single_res=mpi_return_single_res_xyz
                )

                if config.gmnr.TRAIN.use_normalized_xyz:
                    stylegan2_mpi_xyz_input = mpi_tex_pix_normalized_xyz
                else:
                    stylegan2_mpi_xyz_input = mpi_tex_pix_xyz

                if batch_i == 0:
                    logger.info(f"\nmpi_tex_pix_xyz: {stylegan2_mpi_xyz_input[4][:, 0, 0, 2]} \n")

                # Generate images for discriminator training
                with torch.no_grad():

                    real_bs = real_imgs.shape[0]

                    if (config.gmnr.TRAIN.n_view_per_z_in_train > 1) and (
                        config.gmnr.TRAIN.G_select_worse_view == "none"
                    ):
                        # NOTE: if we do not select worst view for G and we use multi-view setup,
                        # to make sure real and fake have same number of views,
                        # we need to reduce the number of generated MPIs.
                        fake_bs = real_bs // config.gmnr.TRAIN.n_view_per_z_in_train
                        assert (
                            real_bs % config.gmnr.TRAIN.n_view_per_z_in_train == 0
                        ), f"{real_bs}, {config.gmnr.TRAIN.n_view_per_z_in_train}"
                    else:
                        fake_bs = real_bs

                    z = z_sampler(
                        (fake_bs, metadata["latent_dim"]),
                        device=device,
                        dist=metadata["z_dist"],
                    )

                    split_batch_size_list = [
                        z.shape[0] // metadata["batch_split"] for _ in range(metadata["batch_split"])
                    ]
                    n_diff = z.shape[0] - np.sum(split_batch_size_list)
                    for i in range(n_diff):
                        split_batch_size_list[i] += 1

                    gen_imgs = []
                    gen_positions = []
                    flat_w2c_mats_gen = []
                    start_batch_idx = 0

                    for split in range(metadata["batch_split"]):
                        split_batch_size = split_batch_size_list[split]
                        subset_z = z[start_batch_idx : start_batch_idx + split_batch_size]
                        start_batch_idx = start_batch_idx + split_batch_size

                        g_imgs, g_pos, g_c2w_mats, _, _ = generator_ddp(
                            z=subset_z,
                            c=None,
                            mpi_xyz_coords=stylegan2_mpi_xyz_input,
                            xyz_coords_only_z=xyz_coords_only_z,
                            n_planes=config.gmnr.MPI.n_gen_planes,
                            mpi_renderer = mpi_renderer,
                            iteration =  discriminator.step,
                            enable_mapping_grad=enable_mapping_grad,
                            enable_syn_feat_net_grad=enable_syn_feat_net_grad,
                            truncation_psi=config.gmnr.MODEL.STYLEGAN2.truncation_psi,
                            horizontal_std = cur_h_stddev,
                            vertical_std=cur_v_stddev,
                            given_yaws=None,
                            given_pitches=None,
                        )


                        tmp_n_view_per_z_in_train = config.gmnr.TRAIN.n_view_per_z_in_train
                        if (config.gmnr.TRAIN.n_view_per_z_in_train > 1) and (
                            config.gmnr.TRAIN.G_select_worse_view != "none"
                        ):
                            # NOTE: we need to make sure D is exposed to same number of real/fake images
                            tmp_n_view_per_z_in_train = 1



                        if config.gmnr.TRAIN.D_cond_pose_dim == 9:
                            g_w2c_mats = torch.inverse(g_c2w_mats[:, :3, :3])
                        elif config.gmnr.TRAIN.D_cond_pose_dim == 16:
                            g_w2c_mats = torch.inverse(g_c2w_mats)
                        else:
                            raise ValueError
                        flat_gen_w2c_mats = g_w2c_mats.reshape((split_batch_size * tmp_n_view_per_z_in_train, -1))
                        flat_w2c_mats_gen.append(flat_gen_w2c_mats)

                        gen_imgs.append(g_imgs)
                        gen_positions.append(g_pos)

                    gen_imgs = torch.cat(gen_imgs, axis=0)
                    gen_positions = torch.cat(gen_positions, axis=0)
                    flat_w2c_mats_gen = torch.cat(flat_w2c_mats_gen, dim=0)

                real_imgs.requires_grad = True
                if config.gmnr.TRAIN.D_cond_on_pose:
                    r_preds, z_preds, pose_preds = discriminator_ddp(real_imgs, alpha, flat_w2c_mats_real, **metadata)
                else:
                    r_preds, z_preds, pose_preds = discriminator_ddp(real_imgs, alpha, None, **metadata)

                if config.gmnr.TRAIN.D_train and metadata["r1_lambda"] > 0:
                    # Gradient penalty
                    grad_real = torch.autograd.grad(
                        outputs=r_preds.sum(),
                        inputs=real_imgs,
                        create_graph=True,
                    )
                    grad_real = [p for p in grad_real][0]

                if config.gmnr.TRAIN.D_train and metadata["r1_lambda"] > 0:
                    grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                    grad_penalty = 0.5 * metadata["r1_lambda"] * grad_penalty
                else:
                    grad_penalty = torch.zeros(1, device=device)

                if batch_i == 0:
                    logger.info(f"[D] grad_penalty: {grad_penalty}\n")

                if config.gmnr.TRAIN.D_cond_on_pose:
                    g_preds, g_pred_latent, g_pred_position = discriminator_ddp(
                        gen_imgs, alpha, flat_w2c_mats_gen, **metadata
                    )
                else:
                    g_preds, g_pred_latent, g_pred_position = discriminator_ddp(gen_imgs, alpha, None, **metadata)

                d_gan_loss_real = torch.nn.functional.softplus(-r_preds).mean()
                d_gan_loss_fake = torch.nn.functional.softplus(g_preds).mean()
                d_gan_loss = d_gan_loss_real + d_gan_loss_fake
                d_loss = d_gan_loss + grad_penalty
                discriminator_losses.append(d_loss.item())

                optimizer_D.zero_grad()
                if config.gmnr.TRAIN.D_train:
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(discriminator_ddp.parameters(), metadata["grad_clip"])
                    optimizer_D.step()

            # TRAIN GENERATOR
            optimizer_G.zero_grad()

            split_batch_size_list = [fake_bs // metadata["batch_split"] for _ in range(metadata["batch_split"])]
            n_diff = fake_bs - np.sum(split_batch_size_list)
            for i in range(n_diff):
                split_batch_size_list[i] += 1

            assert fake_bs % metadata["batch_split"] == 0, f"{fake_bs}, {metadata['batch_split']}"

            for tmp_g_iter in range(config.gmnr.TRAIN.G_iters):

                z = z_sampler(
                    (fake_bs, metadata["latent_dim"]),
                    device=device,
                    dist=metadata["z_dist"],
                )

                if (config.gmnr.TRAIN.n_view_per_z_in_train > 1) and (config.gmnr.TRAIN.G_select_worse_view != "none"):
                    with torch.no_grad():
                        # NOTE: we use Z from worst-view
                        worst_views_yaws, worst_views_pitches, z = find_worst_view_per_z(
                            # device=device,
                            generator_ddp=generator_ddp,
                            discriminator_ddp=discriminator_ddp,
                            mpi_renderer=mpi_renderer,
                            iteration =  discriminator.step,
                            z=z,
                            config=config,
                            metadata=metadata,
                            stylegan2_mpi_xyz_input=stylegan2_mpi_xyz_input,
                            enable_mapping_grad=enable_mapping_grad,
                            enable_syn_feat_net_grad=enable_syn_feat_net_grad,
                            cur_h_stddev=cur_h_stddev,
                            cur_v_stddev=cur_v_stddev,
                            alpha=alpha,
                            xyz_coords_only_z=xyz_coords_only_z,
                            n_planes=config.gmnr.MPI.n_gen_planes,
                            truncation_psi=config.gmnr.MODEL.STYLEGAN2.truncation_psi,
                        )
                else:
                    worst_views_pitches, worst_views_yaws = None, None

                start_batch_idx = 0
                for split in range(metadata["batch_split"]):

                    split_batch_size = split_batch_size_list[split]
                    subset_z = z[start_batch_idx : (start_batch_idx + split_batch_size)]

                    if (config.gmnr.TRAIN.n_view_per_z_in_train > 1) and (
                        config.gmnr.TRAIN.G_select_worse_view != "none"
                    ):
                        tmp_given_yaws = worst_views_yaws[start_batch_idx : (start_batch_idx + split_batch_size), :]
                        tmp_given_pitches = worst_views_pitches[
                            start_batch_idx : (start_batch_idx + split_batch_size), :
                        ]
                    else:
                        tmp_given_yaws = None
                        tmp_given_pitches = None

                    start_batch_idx = start_batch_idx + split_batch_size

                    gen_imgs, g_pos, gen_c2w_mats, _, depth_map = generator_ddp(
                        z=subset_z,
                        c=None,
                        mpi_xyz_coords=stylegan2_mpi_xyz_input,
                        xyz_coords_only_z=xyz_coords_only_z,
                        n_planes=config.gmnr.MPI.n_gen_planes,
                        mpi_renderer = mpi_renderer,
                        iteration =  discriminator.step,
                        enable_mapping_grad=enable_mapping_grad,
                        enable_syn_feat_net_grad=enable_syn_feat_net_grad,
                        truncation_psi=config.gmnr.MODEL.STYLEGAN2.truncation_psi,
                        horizontal_std = cur_h_stddev,
                        vertical_std=cur_v_stddev,
                        given_yaws=tmp_given_yaws,
                        given_pitches=tmp_given_pitches,
                        view_consistency = True,
                    )

                    gen_imgs, gen_imgs_aux = gen_imgs
                    gen_positions, gen_positions_aux = g_pos
                    gen_c2w_mats, gen_c2w_mats_aux = gen_c2w_mats
                    depth_map, depth_map_aux = depth_map

                    device = 'cuda'
                    batch_size = gen_imgs.shape[0]
                    x, y = torch.meshgrid(torch.linspace(-1, 1, metadata["img_size"], device=device),
                                        torch.linspace(-1, 1, metadata["img_size"], device=device))
                    x = x.T.flatten()
                    y = y.T.flatten()
                    z_coord = torch.ones_like(x, device=device) #/ np.tan((2 * math.pi * 12.6 / 360)/2)

                    rays_d_cam = torch.stack([x, y, z_coord], -1)

                    rays_d_cam = torch.stack(batch_size*[rays_d_cam]).to(device)

                    rays_d_cam = rays_d_cam.reshape((batch_size, metadata["img_size"], metadata["img_size"], 3))
                    rays_d_cam = rays_d_cam * depth_map.reshape(batch_size, metadata["img_size"], metadata["img_size"], 1)
                    
                    primary_points_homogeneous = torch.ones((batch_size, metadata["img_size"], metadata["img_size"], 4), device=device)
                    primary_points_homogeneous[:, :, :, :3] = rays_d_cam

                    primary_points_project_to_auxiliary = torch.bmm(torch.inverse(gen_c2w_mats_aux.float()) @ gen_c2w_mats, primary_points_homogeneous.reshape(batch_size, -1, 4).permute(0,2,1)).permute(0, 2, 1).reshape(batch_size, metadata["img_size"], metadata["img_size"], 4)
                    warp_rgb = F.grid_sample(gen_imgs_aux, primary_points_project_to_auxiliary[..., :2], align_corners=True)


                    tmp_use_edge_aware_loss = (
                        config.gmnr.TRAIN.use_edge_aware_loss
                        and discriminator.step >= config.gmnr.TRAIN.edge_aware_loss_start_iter
                    )

                    if config.gmnr.TRAIN.use_cano_reconstruct_loss or tmp_use_edge_aware_loss:
                        raise NotImplementedError
                    else:
                        edge_smooth_loss = torch.zeros(1, device=device)
                        cano_reconstruct_loss = torch.zeros(1, device=device)

                    if (config.gmnr.TRAIN.n_view_per_z_in_train > 1) and (
                        config.gmnr.TRAIN.G_select_worse_view != "none"
                    ):
                        # NOTE: we have already chosen the worst view, no need to render multiple views.
                        tmp_n_view_per_z_in_train = 1
                    else:
                        tmp_n_view_per_z_in_train = config.gmnr.TRAIN.n_view_per_z_in_train

                    # reprojection_loss = 0

                    if True:
                        pred = (warp_rgb + 1) / 2
                        target = (gen_imgs + 1) / 2
                        abs_diff = torch.abs(target - pred)
                        l1_loss = abs_diff.mean(1, True)

                        ssim_loss = ssim(pred, target).mean(1, True)
                        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
                        reprojection_loss = reprojection_loss.mean() * 0.5

                    if config.gmnr.TRAIN.D_cond_pose_dim == 9:
                        gen_w2c_mats = torch.inverse(gen_c2w_mats[:, :3, :3])
                    elif config.gmnr.TRAIN.D_cond_pose_dim == 16:
                        gen_w2c_mats = torch.inverse(gen_c2w_mats)
                    else:
                        raise ValueError
                    flat_gen_w2c_mats = gen_w2c_mats.reshape((split_batch_size * tmp_n_view_per_z_in_train, -1))

                    if config.gmnr.TRAIN.D_cond_on_pose:
                        g_preds, g_pred_latent, g_pred_position = discriminator_ddp(
                            gen_imgs, alpha, flat_gen_w2c_mats, **metadata
                        )
                    else:
                        g_preds, g_pred_latent, g_pred_position = discriminator_ddp(gen_imgs, alpha, None, **metadata)

                    topk_num = int(g_preds.shape[0])

                    # NOTE: -1 * log sigmoid(f(x)) = log (1 + exp(-f(x))) = softplus(-f(x))
                    # Therefore, large g_preds/f(x) indicates small G_loss
                    g_preds = torch.topk(g_preds, topk_num, dim=0, sorted=False).values

                    g_gan_loss = torch.nn.functional.softplus(-g_preds).mean()

                    g_loss = g_gan_loss + edge_smooth_loss + cano_reconstruct_loss + reprojection_loss
                    generator_losses.append(g_loss.item())

                    # NOTE: to make sure loss are same as using a whole batch
                    g_loss = g_loss / (metadata["batch_split"] + 1e-8)

                    g_loss.backward()

                torch.nn.utils.clip_grad_norm_(generator_ddp.parameters(), metadata.get("grad_clip", 0.3))
                optimizer_G.step()

                ema.update(generator_ddp.parameters())
                ema2.update(generator_ddp.parameters())

            if rank == 0:
                if discriminator.step % 10 == 0:
                    tqdm.write(
                        f"[Experiment: {opt.output_dir}] "
                        # f"[Epoch: {discriminator.epoch}/{opt.n_epochs}] "
                        f"[D loss: {d_loss.item()}] [G loss: {g_loss.item()}] "
                        f"[Step: {discriminator.step}] "
                        f"[Img Size: {metadata['img_size']}] [Batch Size: {metadata['batch_size']}] "
                        f"[TopK: {topk_num}] "
                    )

                    # fmt: off
                    writer.add_scalar(f"loss_d", d_gan_loss.item(), global_step=int(discriminator.step))
                    writer.add_scalar(f"loss_d_real", d_gan_loss_real.item(), global_step=int(discriminator.step))
                    writer.add_scalar(f"loss_d_fake", d_gan_loss_fake.item(), global_step=int(discriminator.step))
                    writer.add_scalar(f"loss_g_fake", g_gan_loss.item(), global_step=int(discriminator.step))
                    writer.add_scalar(f"grad_penalty", grad_penalty.item(), global_step=int(discriminator.step))
                    writer.add_scalar(f"edge_smooth_loss", edge_smooth_loss.item(), global_step=int(discriminator.step))
                    writer.add_scalar(f"cano_reconstruct_loss", cano_reconstruct_loss.item(), global_step=int(discriminator.step))
                    writer.add_scalar(f"d_fade_in_alpha", alpha, global_step=int(discriminator.step))
                    writer.add_scalar(f"cur_h_stddev", cur_h_stddev, global_step=int(discriminator.step))
                    writer.add_scalar(f"cur_v_stddev", cur_v_stddev, global_step=int(discriminator.step))
                    # if config.gmnr.TRAIN.aug_with_lighting:
                    #     writer.add_scalar(f"cur_ka", light_renderer.cur_ka, global_step=int(discriminator.step))
                    #     writer.add_scalar(f"cur_kd", light_renderer.cur_kd, global_step=int(discriminator.step))
                    # fmt: on

            # fmt: off
            if (rank == 0):
                # if (discriminator.step > 0) and (discriminator.step % opt.sample_interval == 0):
                if (discriminator.step % opt.sample_interval == 0):
                    mpi_tex_pix_xyz, mpi_tex_pix_normalized_xyz = mpi_renderer.get_xyz(metadata["tex_size"], metadata["tex_size"], ret_single_res=mpi_return_single_res_xyz)
                    if config.gmnr.TRAIN.use_normalized_xyz:
                        stylegan2_mpi_xyz_input = mpi_tex_pix_normalized_xyz
                    else:
                        stylegan2_mpi_xyz_input = mpi_tex_pix_xyz
                    minibatch = 1
                    n_minibatches = int(np.ceil(fixed_z.shape[0] / minibatch))

                    generator_ddp.eval()
                    with torch.no_grad():
                        copied_metadata = copy.deepcopy(metadata)
                        copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                        copied_metadata['img_size'] = copied_metadata['eval_img_size']
                        gen_imgs = []
                        for i in range(n_minibatches):
                            mb_mpi_rgbas, _ = generator_ddp.module.video_gen(
                                z=fixed_z[(i * minibatch):((i + 1) * minibatch)].to(device),
                                c=None,
                                mpi_xyz_coords=stylegan2_mpi_xyz_input,
                                xyz_coords_only_z=xyz_coords_only_z,
                                n_planes=config.gmnr.MPI.n_gen_planes,
                                mpi_renderer = mpi_renderer,
                                # iteration =  discriminator.step,
                                truncation_psi=config.gmnr.MODEL.STYLEGAN2.truncation_psi,
                                horizontal_mean=copied_metadata['h_mean'],
                                horizontal_std=copied_metadata['h_stddev'],
                                vertical_mean=copied_metadata['v_mean'],
                                vertical_std=copied_metadata['v_stddev'],
                                # mode = 'test',
                            )
                            mb_gen_imgs, _, _, _, _, _, _, _ = mpi_renderer.render(
                                mb_mpi_rgbas, copied_metadata['img_size'], copied_metadata['img_size'],
                                horizontal_mean=copied_metadata['h_mean'],
                                horizontal_std=copied_metadata['h_stddev'],
                                vertical_mean=copied_metadata['v_mean'],
                                vertical_std=copied_metadata['v_stddev'],
                            )
                            gen_imgs.append(mb_gen_imgs)
                        gen_imgs = torch.cat(gen_imgs, dim=0)

                    gen_fixed = gen_imgs[:25]
                    save_image(gen_imgs[:25], os.path.join(opt.output_dir, f"{discriminator.step}_fixed.png"), nrow=5, normalize=True)


                    with torch.no_grad():
                        if opt.curriculum == "FFHQ":
                            tmp_h_angle = 0.5
                        elif opt.curriculum == "AFHQCat":
                            tmp_h_angle = 0.3
                        elif opt.curriculum == "MetFaces":
                            tmp_h_angle = 0.5
                        else:
                            raise ValueError

                        copied_metadata = copy.deepcopy(metadata)
                        copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                        copied_metadata['h_mean'] += tmp_h_angle
                        copied_metadata['img_size'] = copied_metadata['eval_img_size']
                        if config.gmnr.MPI.CAM_SETUP.cam_pose_n_truncated_stds * metadata["h_stddev"] > tmp_h_angle:
                            gen_imgs = []
                            for i in range(n_minibatches):
                                mb_mpi_rgbas, _ = generator_ddp.module.video_gen(
                                    z=fixed_z[(i * minibatch):((i + 1) * minibatch)].to(device),
                                    c=None,
                                    mpi_xyz_coords=stylegan2_mpi_xyz_input,
                                    xyz_coords_only_z=xyz_coords_only_z,
                                    n_planes=config.gmnr.MPI.n_gen_planes,
                                    mpi_renderer = mpi_renderer,
                                    # iteration =  discriminator.step,
                                    truncation_psi=config.gmnr.MODEL.STYLEGAN2.truncation_psi,
                                    horizontal_mean=copied_metadata['h_mean'],
                                    horizontal_std=copied_metadata['h_stddev'],
                                    vertical_mean=copied_metadata['v_mean'],
                                    vertical_std=copied_metadata['v_stddev'],
                                    # mode = 'test',
                                )
                                mb_gen_imgs, _, _, _, _, _, _, _ = mpi_renderer.render(
                                    mb_mpi_rgbas, copied_metadata['img_size'], copied_metadata['img_size'],
                                    horizontal_mean=copied_metadata['h_mean'],
                                    horizontal_std=copied_metadata['h_stddev'],
                                    vertical_mean=copied_metadata['v_mean'],
                                    vertical_std=copied_metadata['v_stddev'],
                                )
                                gen_imgs.append(mb_gen_imgs)
                            gen_imgs = torch.cat(gen_imgs, dim=0)
                        else:
                            gen_imgs = torch.zeros((25, 3, copied_metadata['img_size'], copied_metadata['img_size']))

                    gen_tilted = gen_imgs[:25]
                    save_image(gen_imgs[:25], os.path.join(opt.output_dir, f"{discriminator.step}_tilted.png"), nrow=5, normalize=True)


                    ema.store(generator_ddp.parameters())
                    ema.copy_to(generator_ddp.parameters())
                    generator_ddp.eval()
                    with torch.no_grad():
                        copied_metadata = copy.deepcopy(metadata)
                        copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                        copied_metadata['img_size'] = copied_metadata['eval_img_size']
                        gen_imgs = []
                        for i in range(n_minibatches):
                            mb_mpi_rgbas, _ = generator_ddp.module.video_gen(
                                z=fixed_z[(i * minibatch):((i + 1) * minibatch)].to(device),
                                c=None,
                                mpi_xyz_coords=stylegan2_mpi_xyz_input,
                                xyz_coords_only_z=xyz_coords_only_z,
                                n_planes=config.gmnr.MPI.n_gen_planes,
                                mpi_renderer = mpi_renderer,
                                # iteration =  discriminator.step,
                                truncation_psi=config.gmnr.MODEL.STYLEGAN2.truncation_psi,
                                horizontal_mean=copied_metadata['h_mean'],
                                horizontal_std=copied_metadata['h_stddev'],
                                vertical_mean=copied_metadata['v_mean'],
                                vertical_std=copied_metadata['v_stddev'],
                                # mode = 'test',
                            )
                            mb_gen_imgs, _, _, _, _, _, _, _ = mpi_renderer.render(
                                mb_mpi_rgbas, copied_metadata['img_size'], copied_metadata['img_size'],
                                horizontal_mean=copied_metadata['h_mean'],
                                horizontal_std=copied_metadata['h_stddev'],
                                vertical_mean=copied_metadata['v_mean'],
                                vertical_std=copied_metadata['v_stddev'],
                            )
                            gen_imgs.append(mb_gen_imgs)
                        gen_imgs = torch.cat(gen_imgs, dim=0)

                    gen_fixed_ema = gen_imgs[:25]
                    save_image(gen_imgs[:25], os.path.join(opt.output_dir, f"{discriminator.step}_fixed_ema.png"), nrow=5, normalize=True)



                    with torch.no_grad():
                        copied_metadata = copy.deepcopy(metadata)
                        copied_metadata['img_size'] = copied_metadata['eval_img_size']
                        copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                        copied_metadata['psi'] = 0.7
                        rnd_z = torch.randn_like(fixed_z).to(device)
                        gen_imgs = []
                        for i in range(n_minibatches):
                            # [B, #planes, 4, tex_h, tex_w]
                            mb_mpi_rgbas, _ = generator_ddp.module.video_gen(
                                z=rnd_z[(i * minibatch):((i + 1) * minibatch)].to(device),
                                c=None,
                                mpi_xyz_coords=stylegan2_mpi_xyz_input,
                                xyz_coords_only_z=xyz_coords_only_z,
                                n_planes=config.gmnr.MPI.n_gen_planes,
                                mpi_renderer = mpi_renderer,
                                # iteration =  discriminator.step,
                                truncation_psi=config.gmnr.MODEL.STYLEGAN2.truncation_psi,
                                horizontal_mean=copied_metadata['h_mean'],
                                horizontal_std=copied_metadata['h_stddev'],
                                vertical_mean=copied_metadata['v_mean'],
                                vertical_std=copied_metadata['v_stddev'],
                                # mode = 'test',
                            )
                            mb_gen_imgs, _, _, _, _, _, _, _ = mpi_renderer.render(
                                mb_mpi_rgbas, copied_metadata['img_size'], copied_metadata['img_size'],
                                horizontal_mean=copied_metadata['h_mean'],
                                horizontal_std=copied_metadata['h_stddev'],
                                vertical_mean=copied_metadata['v_mean'],
                                vertical_std=copied_metadata['v_stddev'],
                            )
                            gen_imgs.append(mb_gen_imgs)
                        gen_imgs = torch.cat(gen_imgs, dim=0)

                    gen_random = gen_imgs[:25]
                    save_image(gen_imgs[:25], os.path.join(opt.output_dir, f"{discriminator.step}_random.png"), nrow=5, normalize=True)

            

                    if config.gmnr.TRAIN.aug_with_lighting and discriminator.step > config.gmnr.TRAIN.aug_with_lighting_start_iter:
                        tmp_mpi_tex_pix_xyz = mpi_tex_pix_xyz[metadata["tex_size"]]
                        # mb_mpi_rgbas = light_renderer.render(mb_mpi_rgbas, mpi_renderer.static_mpi_plane_dhws, tmp_mpi_tex_pix_xyz)
                    tmp_i = int(np.random.choice(mb_mpi_rgbas.shape[0], size=1)[0])
                    mpi_rgb = mb_mpi_rgbas[tmp_i, :, :3, ...]
                    mpi_alpha = mb_mpi_rgbas[tmp_i, :, 3:, ...]

                    gen_rgb = mpi_rgb
                    gen_alpha = mpi_alpha
                    save_image(mpi_rgb, os.path.join(opt.output_dir, f"{discriminator.step}_random_mpi_rgb.png"), nrow=8, normalize=False)
                    save_image(mpi_alpha, os.path.join(opt.output_dir, f"{discriminator.step}_random_mpi_alpha.png"), nrow=8, normalize=True)

                    


                    ema.restore(generator_ddp.parameters())

                    torch.cuda.empty_cache()

                if discriminator.step % opt.sample_interval == 0:
                    torch.save(ema.state_dict(), os.path.join(config.CHECKPOINT_FOLDER, 'ema.pth'))
                    torch.save(ema2.state_dict(), os.path.join(config.CHECKPOINT_FOLDER, 'ema2.pth'))
                    torch.save(generator_ddp.module.state_dict(), os.path.join(config.CHECKPOINT_FOLDER, 'generator.pth'))
                    torch.save(discriminator_ddp.module.state_dict(), os.path.join(config.CHECKPOINT_FOLDER, 'discriminator.pth'))
                    torch.save(optimizer_G.state_dict(), os.path.join(config.CHECKPOINT_FOLDER, 'optimizer_G.pth'))
                    torch.save(optimizer_D.state_dict(), os.path.join(config.CHECKPOINT_FOLDER, 'optimizer_D.pth'))
                    torch.save(generator_losses, os.path.join(config.CHECKPOINT_FOLDER, 'generator.losses'))
                    torch.save(discriminator_losses, os.path.join(config.CHECKPOINT_FOLDER, 'discriminator.losses'))

                # fmt: on

                with torch.cuda.device(device):
                    torch.cuda.empty_cache()

            discriminator.step += 1
            generator.step += 1

        discriminator.epoch += 1
        generator.epoch += 1

        with torch.cuda.device(device):
            torch.cuda.empty_cache()

    if rank == 0:
        writer.close()

    cleanup()
