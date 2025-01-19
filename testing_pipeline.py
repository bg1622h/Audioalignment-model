from learning_pipeline import AudioDataset, collate_fn, load_checkpoint,AudioAligmentModel, BLANK_CHAR
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchaudio
import torch
import torch.nn as nn
from args_parser import parse_args
def testing_aligment(model, dataloader):
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            audio, target, target_size = batch['audio'], batch['notes'], batch['size']
            output = model(audio)
            output = output.log_softmax(2)
            result = [beam_search_decoder(prob) for prob in output]
            print(result)
def beam_search_decoder(probs, beam_width=3):
    T, C = probs.shape
    sequences = [(([],BLANK_CHAR), 0.0)]
    for t in range(T):
        all_candidates = []
        for (seq, last_char), score in sequences:
            for cur_char in range(C):
                new_seq = seq
                if (cur_char != last_char) and (cur_char != BLANK_CHAR):
                    new_seq = seq + [cur_char]
                new_score = score + probs[t, cur_char].item()
                all_candidates.append(((new_seq, cur_char), new_score))
        sequences = sorted(all_candidates, key=lambda x: -x[1])[:beam_width]
    return sequences

def testing_aligment(model, dataloader):
    model.eval()
    criterion = nn.CTCLoss(blank = 0)
    with torch.no_grad():
        for batch in dataloader:
            audio, target, target_size = batch['audio'], batch['notes'], batch['size']
            output = model(audio)
            output = output.log_softmax(2)
            output_size = torch.tensor([out.size(0) for out in output], dtype=torch.int32, device=device)
            loss = criterion(output.permute(1, 0, 2), target, output_size, target_size)
            print(loss.item())
            result = [beam_search_decoder(prob.cpu()) for prob in output]
            print(result)

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = torchaudio.transforms.Spectrogram(n_fft = args.nfft)

    dataset = AudioDataset(audio_dir=args.audio_dir, midi_dir = args.midi_dir, transform=transform, hop_size=args.nfft // 2)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, len(dataset)))

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    #train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    #train_dataloader = DataLoader (
    #    train_dataset,
    #    batch_size=args.batch_size,
    #    collate_fn=collate_fn,
    #    shuffle=True
    #)

    val_dataloader = DataLoader (
        val_dataset,
        batch_size = 4,
        collate_fn=collate_fn,
    )

    model = AudioAligmentModel(input_dim = args.nfft // 2 + 1, num_classes = 129).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.num_epochs)
    load_checkpoint(model, optimizer, scheduler, "./checkpoints/model50.pth")

    testing_aligment(model, val_dataloader)




    