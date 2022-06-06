import os

import fire
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from retry.api import retry_call
from tqdm import tqdm

from lightweight_gan.diff_augment_test import DiffAugmentTest
from lightweight_gan.exceptions import NanException
from lightweight_gan.trainer import Trainer
from lightweight_gan.utils import cast_list, current_iso_datetime, default, set_seed


def run_training(
    rank,
    world_size,
    model_args,
    data,
    load_from,
    new,
    num_train_steps,
    name,
    seed,
    use_aim,
    aim_repo,
    aim_run_hash,
):
    """run training job

    Args:
        rank: int, id of process
        world_size: int, number of processes participating in training
        model_args: dict, arguments for model
        data: dict, arguments for data
        load_from: str, path to load model from
        new: bool, whether to create new model
        num_train_steps: int, number of training steps
        name: str, name of training
        seed: int, seed for random number generator
        use_aim: bool, whether to use AIM
        aim_repo: str, path to AIM repository
        aim_run_hash: str, hash of AIM run

    Returns:
        None
    """

    is_main = rank == 0
    is_ddp = world_size > 1

    if is_ddp:
        set_seed(seed)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        print(f"{rank + 1}/{world_size} process initialized.")

    model_args.update(is_ddp=is_ddp, rank=rank, world_size=world_size)

    model = Trainer(
        **model_args,
        hparams=model_args,
        use_aim=use_aim,
        aim_repo=aim_repo,
        aim_run_hash=aim_run_hash,
    )

    if not new:
        model.load(load_from)
    else:
        model.clear()

    model.set_data_src(data)

    progress_bar = tqdm(
        initial=model.steps,
        total=num_train_steps,
        mininterval=10.0,
        desc=f"{name}<{data}>",
    )
    while model.steps < num_train_steps:
        retry_call(model.train, tries=3, exceptions=NanException)
        progress_bar.n = model.steps
        progress_bar.refresh()
        if is_main and model.steps % 50 == 0:
            model.print_log()

    model.save(model.checkpoint_num)

    if is_ddp:
        dist.destroy_process_group()


def train_from_folder(
    data="./data",
    results_dir="./results",
    models_dir="./models",
    name="default",
    new=False,
    load_from=-1,
    image_size=256,
    optimizer="adam",
    fmap_max=512,
    transparent=False,
    greyscale=False,
    batch_size=10,
    gradient_accumulate_every=4,
    num_train_steps=150000,
    learning_rate=2e-4,
    save_every=1000,
    evaluate_every=1000,
    generate=False,
    generate_types=["default", "ema"],
    generate_interpolation=False,
    aug_test=False,
    aug_prob=None,
    aug_types=["cutout", "translation"],
    dataset_aug_prob=0.0,
    attn_res_layers=[32],
    freq_chan_attn=False,
    disc_output_size=1,
    dual_contrast_loss=False,
    antialias=False,
    interpolation_num_steps=100,
    save_frames=False,
    num_image_tiles=None,
    num_workers=None,
    multi_gpus=False,
    calculate_fid_every=None,
    calculate_fid_num_images=12800,
    clear_fid_cache=False,
    seed=42,
    amp=False,
    show_progress=False,
    use_aim=False,
    aim_repo=None,
    aim_run_hash=None,
    load_strict=True,
):
    """available commands:
        generate - generate images
        generate_interpolation - generate images with interpolation
        aug_test - test augmentation
        else: train

    Args:
        -- actual args for train_from_folder --
        -- generate command --
        generate: bool, whether to generate
        generate_types: list, types of generation

        -- generate_interpolation command --
        generate_interpolation: bool, whether to use interpolation
        interpolation_num_steps: int, number of interpolation steps
        save_frames: bool, whether to save frames

        -- aug_test command --
        aug_test: bool, whether to use augmentation test

        -- show_progress command --
        show_progress: bool, whether to show progress

        multi_gpus: bool, whether to use multi gpu

        -- run_training args --
        data: str, path to data
        new: bool, whether to create new model
        load_from: int, load model from this checkpoint
        num_train_steps: int, number of training steps
        seed: int, seed for random number generator
        use_aim: bool, whether to use AIM
        aim_repo: str, path to AIM repository
        aim_run_hash: str, hash of AIM run

        -- model args --
        results_dir: str, path to results
        models_dir: str, path to models
        name: str, name of training
        image_size: int, image size
        optimizer: str, optimizer
        fmap_max: int, fmap max
        transparent: bool, whether to use transparent
        greyscale: bool, whether to use greyscale
        batch_size: int, batch size
        gradient_accumulate_every: int, gradient accumulate every
        learning_rate: float, learning rate
        save_every: int, save every
        evaluate_every: int, evaluate every
        aug_prob: float, probability of augmentation
        aug_types: list, types of augmentation
        dataset_aug_prob: float, probability of dataset augmentation
        attn_res_layers: list, attention resolution layers
        freq_chan_attn: bool, whether to use frequency channel attention
        disc_output_size: int, discriminator output size
        dual_contrast_loss: bool, whether to use dual contrast loss
        antialias: bool, whether to use antialias
        num_image_tiles: int, number of image tiles
        num_workers: int, number of workers
        calculate_fid_every: int, calculate fid every
        calculate_fid_num_images: int, number of images to calculate fid
        clear_fid_cache: bool, whether to clear fid cache
        amp: bool, whether to use amp
        load_strict: bool, whether to load strict

    Returns:
        None
    """

    assert (
        torch.cuda.is_available()
    ), "You need to have an Nvidia GPU with CUDA installed."

    num_image_tiles = default(num_image_tiles, 4 if image_size > 512 else 8)

    model_args = dict(
        name=name,
        results_dir=results_dir,
        models_dir=models_dir,
        batch_size=batch_size,
        gradient_accumulate_every=gradient_accumulate_every,
        attn_res_layers=cast_list(attn_res_layers),
        freq_chan_attn=freq_chan_attn,
        disc_output_size=disc_output_size,
        dual_contrast_loss=dual_contrast_loss,
        antialias=antialias,
        image_size=image_size,
        num_image_tiles=num_image_tiles,
        optimizer=optimizer,
        num_workers=num_workers,
        fmap_max=fmap_max,
        transparent=transparent,
        greyscale=greyscale,
        lr=learning_rate,
        save_every=save_every,
        evaluate_every=evaluate_every,
        aug_prob=aug_prob,
        aug_types=cast_list(aug_types),
        dataset_aug_prob=dataset_aug_prob,
        calculate_fid_every=calculate_fid_every,
        calculate_fid_num_images=calculate_fid_num_images,
        clear_fid_cache=clear_fid_cache,
        amp=amp,
        load_strict=load_strict,
    )

    if generate:
        model = Trainer(**model_args)
        model.load(load_from)
        samples_name = current_iso_datetime()
        checkpoint = model.checkpoint_num
        dir_result = model.generate(
            samples_name, num_image_tiles, checkpoint, generate_types
        )
        print(f"sample images generated at {dir_result}")
        return

    if generate_interpolation:
        model = Trainer(**model_args)
        model.load(load_from)
        samples_name = current_iso_datetime()
        model.generate_interpolation(
            samples_name,
            num_image_tiles,
            num_steps=interpolation_num_steps,
            save_frames=save_frames,
        )
        print(f"interpolation generated at {results_dir}/{name}/{samples_name}")
        return

    if show_progress:
        model = Trainer(**model_args)
        model.show_progress(num_images=num_image_tiles, types=generate_types)
        return

    if aug_test:
        DiffAugmentTest(
            data=data,
            image_size=image_size,
            batch_size=batch_size,
            types=aug_types,
            nrow=num_image_tiles,
        )
        return

    world_size = torch.cuda.device_count()

    if world_size == 1 or not multi_gpus:
        run_training(
            0,
            1,
            model_args,
            data,
            load_from,
            new,
            num_train_steps,
            name,
            seed,
            use_aim,
            aim_repo,
            aim_run_hash,
        )
    else:
        mp.spawn(
            run_training,
            args=(
                world_size,
                model_args,
                data,
                load_from,
                new,
                num_train_steps,
                name,
                seed,
                use_aim,
                aim_repo,
                aim_run_hash,
            ),
            nprocs=world_size,
            join=True,
        )


def main():
    fire.Fire(train_from_folder)
