from models.custom import Generator
from get_level import GetLevel as getLevel
from data_loader import MarioDataset
from image_gen.image_gen import GameImageGenerator
from image_gen.fixer import PipeFixer
from image_gen.asset_map import get_asset_map
from cursesmenu import CursesMenu, SelectionMenu
import numpy as np
import torch
import os

'''
Generates levels from different conditioned generators
'''

if __name__ == "__main__":
    # Initialize dataset and generator
    conditional_channels = [
        0,
        1,
        2,
        5,
        6,
        7,
    ]  # channels on which generator is conditioned on
    dataset = MarioDataset()
    netG = Generator(
        latent_size=(len(conditional_channels) + 1, 14, 14), out_size=(13, 32, 32)
    )
    netG.load_state_dict(torch.load(
        "trained_models/netG_epoch_condition_012567_sky_390000_0_32.pth"))
    mario_map = get_asset_map(game="mario")
    gen = GameImageGenerator(asset_map=mario_map)
    prev_frame, curr_frame = dataset[[20]]  # 51
    fixer = PipeFixer()
    level_gen = getLevel(netG, gen, fixer, prev_frame,
                         curr_frame, conditional_channels)


    model_opt = (
        "trained_models/netG_epoch_condition_012567_sky_390000_0_32.pth",
        "trained_models/netG_epoch_condition_012567_sky_390000_0_32 copy.pth",
        "trained_models/netG_epoch_condition_012567_enemy_390000_0_32.pth",
        "trained_models/netG_epoch_condition_012567_310000_0_32.pth",
        )
    
    # Create menu for conditional feature selection
    features = ["More Sky Tiles", "Underground", "More Enemies", "Random"]
    file_name = "my_GAN_level_random"
    while True:
        selection = SelectionMenu.get_selection(
            features, title="Select the features you would like to generate:",
        )
        var = 0.2
        level_gen.conditional_channels = conditional_channels
        try:
            model = model_opt[selection]
            noise = np.zeros((196,))
            level_gen.netG.load_state_dict(torch.load(model))
            level = level_gen.generate_frames(noise, var=var, frame_count=1)
            level_gen.save_full_level_pkl(file_name)
            level_gen.save_full_level_rom(file_name)
            level_gen.save_full_level_np(file_name)
            level_gen.gen.save_gen_level(img_name=file_name) # this saves images
        except IndexError:
            break
