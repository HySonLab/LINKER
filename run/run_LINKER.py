import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.modules import *
from model.loss import *
from dataloader.dataloader import *
import argparse

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for idx, batch in enumerate(dataloader):
        # Move to device
        names, prot_tensors, prot_masks, batched_graph, fg_indices_tensor, fg_type_tensor, padded_labels, padded_label_masks, values = batch
        prot_tensors = prot_tensors.to(device)
        prot_masks = prot_masks.to(device)
        batched_graph = batched_graph.to(device)
        fg_indices_tensor = fg_indices_tensor.to(device)
        fg_type_tensor = fg_type_tensor.to(device)
        padded_labels = padded_labels.to(device)
        padded_label_masks = padded_label_masks.to(device)

        optimizer.zero_grad()
        logits = model(prot_tensors, prot_masks, batched_graph, fg_indices_tensor, fg_type_tensor)  # [B, F, R, C]

        # Align logits and labels shape: [B, F, R, C] → permute to match labels if needed
        loss = criterion(logits, padded_labels)  # shape [B,F,R,C]

        loss = loss * padded_label_masks.unsqueeze(-1)  # Apply mask
        loss = loss.sum() / (padded_label_masks.sum() * padded_labels.size(-1) + 1e-6)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        step_log = f"Step {idx+1}: Train Loss = {loss.item():.6f}"
        print(step_log)
    return total_loss / len(dataloader)

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            names, prot_tensors, prot_masks, batched_graph, fg_indices_tensor, fg_type_tensor, padded_labels, padded_label_masks, values = batch
            prot_tensors = prot_tensors.to(device)
            prot_masks = prot_masks.to(device)
            batched_graph = batched_graph.to(device)
            fg_indices_tensor = fg_indices_tensor.to(device)
            fg_type_tensor = fg_type_tensor.to(device)
            padded_labels = padded_labels.to(device)
            padded_label_masks = padded_label_masks.to(device)

            logits = model(prot_tensors, prot_masks, batched_graph, fg_indices_tensor, fg_type_tensor)
            loss = criterion(logits, padded_labels)
            loss = loss * padded_label_masks.unsqueeze(-1)
            loss = loss.sum() / (padded_label_masks.sum() * padded_labels.size(-1) + 1e-6)

            total_loss += loss.item()
            step_log = f"Step {idx+1}: Val Loss = {loss.item():.6f}"
            print(step_log)
    return total_loss / len(dataloader)

def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=30, save_path='best_model.pt'):
    best_val_loss = float('inf')
    best_epoch = -1

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
            print(f"✔️ Saved new best model at epoch {epoch+1} with val loss {val_loss:.4f}")

    print(f"✅ Training complete. Best epoch: {best_epoch+1} with Val Loss: {best_val_loss:.4f}")

def test_and_save_logits(model, test_loader, device, save_dir='./predictions'):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            protein_names, prot_tensors, prot_masks, batched_graph, fg_indices_tensor, fg_type_tensor, padded_labels, padded_label_masks, values = batch
            print(protein_names)
            prot_tensors = prot_tensors.to(device)
            prot_masks = prot_masks.to(device)
            batched_graph = batched_graph.to(device)
            fg_indices_tensor = fg_indices_tensor.to(device)
            fg_type_tensor = fg_type_tensor.to(device)
            padded_labels = padded_labels.to(device)
            padded_label_masks = padded_label_masks.to(device)

            logits = model(prot_tensors, prot_masks, batched_graph, fg_indices_tensor, fg_type_tensor)

            # Với từng sample trong batch
            for i, name in enumerate(protein_names):
                output = {
                    'logits': logits[i].cpu(),   # [R, F, C]
                    'label': padded_labels[i].cpu(),  # [R, F, C]
                    'mask': padded_label_masks[i].cpu(),  # [R, F]
                }
                torch.save(output, os.path.join(save_dir, f'{name}.pt'))

                # Nếu bạn muốn lưu thêm bản text dễ đọc:
                with open(os.path.join(save_dir, f'{name}.txt'), 'w') as f:
                    f.write(f'Logits:\n{output["logits"]}\n\n')
                    f.write(f'Label:\n{output["label"]}\n\n')
                    f.write(f'Mask:\n{output["mask"]}\n')

    print(f'Logits saved to: {save_dir}')



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--protein_emb_path", type=str, required=True)
    parser.add_argument("--fg_instance_path", type=str, required=True)
    parser.add_argument("--ligand_graph_path", type=str, required=True)
    parser.add_argument("--label_path", type=str, required=True)
    return parser.parse_args()



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    

    train_dataset                   = LINKER_Dataloader(args.csv_path, 'train', args.protein_emb_path, args.fg_instance_path, args.ligand_graph_path, args.label_path)
    val_dataset                     = LINKER_Dataloader(args.csv_path, 'val',   args.protein_emb_path, args.fg_instance_path, args.ligand_graph_path, args.label_path)
    test_dataset                    = LINKER_Dataloader(args.csv_path, 'test',  args.protein_emb_path, args.fg_instance_path, args.ligand_graph_path, args.label_path)

    train_loader                    = DataLoader(train_dataset,  batch_size=2, shuffle=False,  collate_fn=collate_fn_LINKER)
    val_loader                      = DataLoader(val_dataset,  batch_size=2, shuffle=False,  collate_fn=collate_fn_LINKER)
    test_loader                     = DataLoader(test_dataset,  batch_size=2, shuffle=False,  collate_fn=collate_fn_LINKER)


    model                           = LINKER().to(device)
    
    # state_dict = torch.load('best_model.pt', map_location=device)
    # model.load_state_dict(state_dict)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = FocalLoss(alpha=0.85, gamma=1, reduction='none')

    train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=20, save_path='checkpoints/best_model.pt')
    test_and_save_logits(model, test_loader, device, save_dir='evaluation/best_logits')