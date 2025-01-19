import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Audio-MIDI Dataset Pipeline")
    parser.add_argument("--audio_dir", type = str, default="dataset", help = "The path to the audio files folder")
    parser.add_argument("--midi_dir", type = str, default="dataset", help = "The path to the folder with MIDI files")
    parser.add_argument("--batch_size", type = int, default = 64)
    parser.add_argument("--lr", type = float, default=0.01)
    parser.add_argument("--num_epochs", type = int, default = 5)
    parser.add_argument("--nfft", type = int, default = 400)
    parser.add_argument("--taskname", type = str, default="Testing training script")
    parser.add_argument("--env_path", type = str, default="settings.env")
    parser.add_argument("--load_path", type = str, default = None)
    parser.add_argument("--save_path_template", type = str, default="checkpoints/")
    parser.add_argument("--save_step", type = int, default = 5)
    parser.add_argument("--clearml_reuse", type = bool, default = True)
    parser.add_argument("--train_part", type = float, default=0.8)
    parser.add_argument("--dataset_size", type = int, default=50)
    return parser.parse_args()