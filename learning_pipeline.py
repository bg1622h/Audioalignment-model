import os
import torchaudio
import torch
import pretty_midi
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from args_parser import parse_args
from clearml import Task, Logger
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import soundfile as sf #saving sound files
from sklearn.metrics.pairwise import cosine_similarity
PITCHES = 128
SAMPLE_RATE = 44100
BLANK_CHAR = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Error(Exception):
    pass

def midi_processing(midi_data):
    data = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            data.append([note.pitch + 1, note.start, note.end, note.velocity])
    return data

def create_midi_template():
    midi = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    start = 0.5
    notes = []
    for i in range(PITCHES):
        notes.append((i, start, 0.5))
        start+=0.5
    velocity = 100
    for note, start, duration in notes:
        midi_note = pretty_midi.Note(velocity=velocity, pitch=note, start=start, end=start + duration)
        piano.notes.append(midi_note)
    midi.instruments.append(piano)
    return midi

#transform - resampling + spectrogram
class AudioDataset(Dataset):
    """
    Load files for every batch from disk to save RAM
    """
    def __init__(self,audio_dir,midi_dir,hop_size, transform = None, new_sr = None):
        self.audio_dir = audio_dir
        self.midi_dir = midi_dir
        self.new_sr = new_sr
        self.hop_size = hop_size
        self.transform = transform
        self.audio_files = [f for f in os.listdir(audio_dir) if f.endswith((".wav", ".mp3"))]
        self.midi_files = [f for f in os.listdir(midi_dir) if f.endswith((".midi"))]
        assert len(self.audio_files) == len(self.midi_files), f"Audio files count: {len(self.audio_files)} but Midi files count: {len(self.midi_files)}"
        self.audio_files.sort()
        self.midi_files.sort()
        for audio_file, midi_file in zip(self.audio_files, self.midi_files):
            name_audio = os.path.splitext(audio_file)[0]
            name_midi = os.path.splitext(midi_file)[0]
            if name_audio < name_midi:
                raise Error(f"{name_audio}.midi file not found")
            if name_audio > name_midi:
                raise Error(f"{name_midi}.* audio file not found")
            
    def __len__(self):
        return len(self.audio_files)     
    def __getitem__(self, index):
        audio, sr = torchaudio.load(os.path.join(self.audio_dir, self.audio_files[index]))
        audio = audio.mean(dim = 0)
        if self.transform:
            audio = self.transform(audio)
            if self.new_sr:
                sr = self.new_sr
        midi_data = midi_processing(pretty_midi.PrettyMIDI(os.path.join(self.midi_dir, self.midi_files[index])))
        self.input_size = audio.size()
        audio_segments = [audio[:, i:i + 2 * (sr + self.hop_size - 1) // self.hop_size] for i in range(0,audio.size(1), 2 * (sr + self.hop_size - 1) // self.hop_size)]
        if (audio_segments[-1].size(1) != audio_segments[0].size(1)):
            audio_segments[-1] = torch.cat((audio_segments[-1],torch.zeros(audio_segments[0].size(0), audio_segments[0].size(1) - audio_segments[-1].size(1))), dim = 1)
        segment = []
        for i, audio_segment in enumerate(audio_segments):
            notes = [data[0] for data in midi_data
                     if data[1] >= 2 * i and data[2] < 2 * (i + 1)]
            segment.append({
                'audio': audio_segment,
                'notes': notes,
                'size': len(notes),
                'sr': sr 
            })
        return segment

def collate_fn(batch):
    audio = [item['audio'] for segment in batch for item in segment]
    notes = [torch.tensor(item['notes'], dtype=torch.int32) for segment in batch for item in segment]
    size = [item['size'] for segment in batch for item in segment]
    sr = [item['sr'] for segment in batch for item in segment]
    audio = torch.stack(audio).to(device)
    notes = nn.utils.rnn.pad_sequence(notes, batch_first=True, padding_value=0).to(device)
    return {
        'audio': audio,
        'notes': notes,
        'size': torch.tensor(size, dtype=torch.int32, device=device),
        'sr': torch.tensor(sr, dtype=torch.int32, device=device)
    }

class AudioAligmentModel(nn.Module):
    def __init__(self,output_dim,input_dim, num_transformer_layers = 2, d_model = 256, nhead = 2):
        super(AudioAligmentModel,self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, stride=2)
        self.conv2 = nn.Conv1d(64, 256, kernel_size=3, stride=2)
        transformer_layer = nn.TransformerEncoderLayer (
            d_model = d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.25,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_transformer_layers)
        self.fc = nn.Linear(d_model, output_dim)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.transformer(x.permute(0,2,1))
        x = self.fc(x)
        return x
def save_checkpoint(model, optimizer, scheduler, epoch, iteration, path):
    torch.save({
        'epoch': epoch,
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path)
def load_checkpoint(model, optimizer, scheduler, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['iteration']

def melForward(f):
    return 2595 * np.log10(1 + f/700)
def melInverse(m):
    return (10**(m/2595) - 1) * 700

def data_example(loader, cnt, nfft, hopSize, sr, title, series, logger):
    cur_cnt = 0
    for batch in loader:
        if cur_cnt == cnt:
            return
        audio = batch['audio'][0]
        f,ax = plt.subplots(figsize=(26, 7))

        tGrid = np.arange(0,audio.shape[-1]*hopSize, hopSize)/sr
        freqsR=np.arange(0,nfft//2 + 0.001)/nfft * sr
        fGrid = freqsR
        tt,ff = np.meshgrid(tGrid,fGrid)

        im=ax.pcolormesh(tt,ff,20*torch.log10(torch.clamp(audio, min = 1e-300)),cmap="gist_heat")
        ax.set_xlabel('Time, sec', size=20)
        ax.set_ylabel('Frequency, Hz', size=20)
        ax.set_yscale("function",functions=(melForward,melInverse))
        f.colorbar(im)
        fig = plt.gcf()
        canvas = FigureCanvas(fig)
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(canvas.get_width_height()[::-1] + (3,))
        logger.report_image(
            title = title,
            series = f"{series}{cur_cnt}",
            image = image,
        )
        plt.close()
        cur_cnt+=1

def train_model(model, train_dataloader, val_dataloader, num_epochs, initial_lr, logger, checkpoint_path = None, save_checkpoint_path = None, save_step = None):
    model = model.to(device)
    criterion = nn.CTCLoss(blank = 0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = initial_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=(num_epochs) * len(train_dataloader))
    iteration = -1
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        start_epoch, iteration = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
    for epoch in range(start_epoch, num_epochs):
        model.train()
        for batch in train_dataloader:
            iteration+=1
            if save_checkpoint_path and iteration % save_step == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, iteration, save_checkpoint_path+f"model{iteration}.pth")
            audio, target, target_size = batch['audio'], batch['notes'], batch['size']
            output = model(audio)
            output = output.log_softmax(2)
            output_size = torch.tensor([out.size(0) for out in output], dtype = torch.int32, device=device)
            loss = criterion(output.permute(1,0,2), target, output_size, target_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.report_scalar("Loss", "Training Loss", loss.item(), iteration)     
            scheduler.step()
            logger.report_scalar("Learning rate", "Learning rate", scheduler.get_last_lr()[0], iteration)
        model.eval()
        val_loss = 0.0
        val_iterations = 0
        with torch.no_grad():
            for batch in val_dataloader:
                audio, target, target_size = batch['audio'], batch['notes'], batch['size']
                output = model(audio)
                output = output.log_softmax(2)
                output_size = torch.tensor([out.size(0) for out in output], dtype=torch.int32, device=device)
                loss = criterion(output.permute(1, 0, 2), target, output_size, target_size)
                val_loss += loss.item()
                val_iterations += 1
        logger.report_scalar("Loss", "Testing Loss", val_loss / val_iterations if val_iterations > 0 else 0.0, iteration)
    save_checkpoint(model, optimizer, scheduler, epoch, iteration, save_checkpoint_path+f"final_model.pth")

if __name__ == "__main__":
    args = parse_args()
    load_dotenv(args.env_path)

    #midi_template = create_midi_template()
    #midi_template = midi_template.fluidsynth(fs = 44100)
    #sf.write("template.wav", midi_template, 44100)
    #print(midi_template.size())

    task = Task.init(project_name="Audio Aligment", task_name=args.taskname, reuse_last_task_id=args.clearml_reuse)
    transform = torchaudio.transforms.Spectrogram(n_fft = args.nfft)

    dataset = AudioDataset(audio_dir=args.audio_dir, midi_dir = args.midi_dir, transform=transform, hop_size=args.nfft // 2)
    train_size = int(args.train_part * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader (
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True
    )

    val_dataloader = DataLoader (
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True
    )

    model = AudioAligmentModel(input_dim = args.nfft // 2 + 1, output_dim = 129)

    logger = Logger.current_logger()

    #data_example(train_dataloader, 5, args.nfft,args.nfft // 2, 44100, "Train Spectrograms", "Signal", logger)
    #data_example(val_dataloader, 5, args.nfft ,args.nfft // 2, 44100, "Testing Spectrograms", "Signal", logger)

    train_model(model,train_dataloader,val_dataloader,args.num_epochs,args.lr,logger, args.load_path, args.save_path_template, args.save_step)
    Task.current_task().close()