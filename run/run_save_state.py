import sys
import os
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
from model.modules import *
from model.loss import *
from dataloader.dataloader import *
import argparse


def save_all_features_by_protein(
    protein_names,
    prot_tensors,
    fg_embedded,
    fg_mask,
    prot_ctx,
    ligand_ctx,
    protein_output_ark,
    ligand_output_ark,
    prot_mask,
    logits,
    values,
    save_dir
):
    os.makedirs(save_dir, exist_ok=True)

    B = len(protein_names)
    for i in range(B):
        name = protein_names[i]

        # Lấy độ dài thực tế
        prot_len = int(prot_mask[i].sum().item())
        lig_len = int(fg_mask[i].sum().item())

        protein_features = {
            "protein_name": name,
            "prot_tensors": prot_tensors[i, :prot_len].detach().cpu(),
            "fg_embedded": fg_embedded[i, :lig_len].detach().cpu(),
            "fg_mask": fg_mask[i, :lig_len].detach().cpu(),
            "prot_ctx": prot_ctx[i, :prot_len].detach().cpu(),
            "ligand_ctx": ligand_ctx[i, :lig_len].detach().cpu(),
            "protein_output_ark": protein_output_ark[i, :prot_len].detach().cpu(),
            "ligand_output_ark": ligand_output_ark[i, :lig_len].detach().cpu(),
            "prot_mask": prot_mask[i, :prot_len].detach().cpu(),
            "logits": logits[i, :prot_len, :lig_len].detach().cpu(),  # [R, F, 7]
            "value": values[i].squeeze().item()
        }

        save_path = os.path.join(save_dir, f"{name}.pt")
        print(f'Save {save_path}')
        torch.save(protein_features, save_path)

def extract_and_save_features(model, dataloader, device, save_dir: str = "./features"):
    model.eval()
    model.to(device)

    with torch.no_grad():
        for batch in dataloader:
            
            protein_names, prot_tensors, prot_masks, batched_graph, fg_indices_tensor, fg_type_tensor, padded_labels, padded_label_masks, values = batch
            
            prot_tensors = prot_tensors.to(device)
            prot_mask = prot_masks.to(device)
            batched_graph = batched_graph.to(device)
            fg_indices_tensor = fg_indices_tensor.to(device)
            fg_type_tensor = fg_type_tensor.to(device)
            padded_labels = padded_labels.to(device)
            padded_label_masks = padded_label_masks.to(device)
            values = values.to(device)

            # Forward pass (modify if your model returns more)
            fg_embedded, fg_mask = model.finger_id(batched_graph, fg_type_tensor, fg_indices_tensor)
            prot_ctx = model.scat.prot_self_attn(prot_tensors, src_key_padding_mask=~prot_mask.bool())
            ligand_ctx = model.scat.ligand_self_attn(fg_embedded, src_key_padding_mask=~fg_mask.bool())
            prot_ctx = prot_ctx.masked_fill(~prot_mask.unsqueeze(-1).bool(), 0.0)
            ligand_ctx = ligand_ctx.masked_fill(~fg_mask.unsqueeze(-1).bool(), 0.0)

            protein_output_ark, protein_mask_ark, _ = model.scat.arkmab(prot_ctx, prot_mask.float(), ligand_ctx, fg_mask.float())
            ligand_output_ark, ligand_mask_ark, _ = model.scat.arkmab(ligand_ctx, fg_mask.float(), prot_ctx, prot_mask.float())

            
            
            masked_output_protein = protein_output_ark * protein_mask_ark.unsqueeze(-1)  # [B, Lp, D]
            masked_output_ligand = ligand_output_ark* ligand_mask_ark.unsqueeze(-1)  # [B, Lp, D]
            
            logits = model.unet_pair(masked_output_protein, masked_output_ligand)
            # Save per-protein
            save_all_features_by_protein(
                protein_names=protein_names,
                prot_tensors = prot_tensors,
                fg_embedded=fg_embedded,
                fg_mask=fg_mask,
                prot_ctx=prot_ctx,
                ligand_ctx=ligand_ctx,
                protein_output_ark=protein_output_ark,
                ligand_output_ark=ligand_output_ark,
                prot_mask=prot_mask,
                logits = logits,
                values=values,
                save_dir=save_dir
            )
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--protein_emb_path", type=str, required=True)
    parser.add_argument("--fg_instance_path", type=str, required=True)
    parser.add_argument("--ligand_graph_path", type=str, required=True)
    parser.add_argument("--label_path", type=str, required=True)
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt", type=str, required=True)
    parser.add_argument("--save_dir", default="evaluation/extracted_features", type=str, required=True)
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


    model = LINKER().to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    
    extract_and_save_features(model, train_loader, device=device, save_dir=os.path.join(args.save_dir,"train"))
    extract_and_save_features(model, val_loader, device=device, save_dir=os.path.join(args.save_dir,"val"))
    extract_and_save_features(model, test_loader, device=device, save_dir=os.path.join(args.save_dir,"test"))