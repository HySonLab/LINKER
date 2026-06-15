import sys

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.modules import *
from model.loss import *
from dataloader.dta_dataloader import *
import argparse
from datetime import datetime
from tqdm import tqdm
from scipy.stats import pearsonr
from lifelines.utils import concordance_index
from sklearn.metrics import r2_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--num_seeds", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--batch_size", type=str, required=True)
    parser.add_argument("--protein_emb_path", type=str, required=True)
    parser.add_argument("--fg_instance_path", type=str, required=True)
    parser.add_argument("--ligand_graph_path", type=str, required=True)
    return parser.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for newer pytorch versions
    torch.use_deterministic_algorithms(True, warn_only=True)

    os.environ["PYTHONHASHSEED"] = str(seed)



def evaluate(model, loader, device, criterion):
    model.eval()

    val_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for prot_tensors, prot_masks, batched_graph, fg_indices_tensor, fg_type_tensor, labels in loader:

            prot_tensors = prot_tensors.to(device)
            prot_masks = prot_masks.to(device)
            batched_graph = batched_graph.to(device)
            fg_indices_tensor = fg_indices_tensor.to(device)
            fg_type_tensor = fg_type_tensor.to(device)
            labels = labels.to(device).float()

            preds = model(prot_tensors, prot_masks, batched_graph, fg_indices_tensor, fg_type_tensor)
            preds = preds.view(-1)

            loss = criterion(preds, labels)
            val_loss += loss.item()

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())


    all_preds = torch.concatenate(all_preds).reshape(-1, 1)
    all_labels = torch.concatenate(all_labels).reshape(-1, 1)

    rmse = torch.sqrt(((all_preds - all_labels) ** 2).mean())
    avg_loss = val_loss / (len(loader) + 1e-9)
    corr, p_value = pearsonr(all_preds, all_labels)
    ci = concordance_index(all_labels, all_preds)
    r2 = r2_score(
        all_preds.numpy(),
        all_labels.numpy()
    )
    return avg_loss, rmse.item(), corr[0], ci, r2

def test(model, loader, device):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for prot_tensors, prot_masks, batched_graph, fg_indices_tensor, fg_type_tensor, labels in loader:

            prot_tensors = prot_tensors.to(device)
            prot_masks = prot_masks.to(device)
            batched_graph = batched_graph.to(device)
            fg_indices_tensor = fg_indices_tensor.to(device)
            fg_type_tensor = fg_type_tensor.to(device)
            labels = labels.to(device).float()

            preds = model(prot_tensors, prot_masks, batched_graph, fg_indices_tensor, fg_type_tensor)
            preds = preds.view(-1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.concatenate(all_preds).flatten()
    all_labels = torch.concatenate(all_labels).flatten()

    rmse = torch.sqrt(((all_preds - all_labels) ** 2).mean())
    corr, p_value = pearsonr(all_preds, all_labels)
    ci = concordance_index(all_labels, all_preds)
    return rmse.item(), corr[0], ci

def test_ensemble(model_paths, loader, device):

    all_fold_preds = []

    for path in model_paths:
        model = DTA_Predictor().to(device)

        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        model.eval()

        fold_preds = []
        fold_labels = []

        with torch.no_grad():
            for prot_tensors, prot_masks, batched_graph, fg_indices_tensor, fg_type_tensor, labels in loader:

                prot_tensors = prot_tensors.to(device)
                prot_masks = prot_masks.to(device)
                batched_graph = batched_graph.to(device)
                fg_indices_tensor = fg_indices_tensor.to(device)
                fg_type_tensor = fg_type_tensor.to(device)
                labels = labels.to(device).float()

                preds = model(prot_tensors, prot_masks, batched_graph, fg_indices_tensor, fg_type_tensor)
                preds = preds.view(-1)
                fold_preds.append(preds.cpu())
                fold_labels.append(labels.cpu())

        fold_preds = torch.concatenate(fold_preds).flatten()
        fold_labels = torch.concatenate(fold_labels).flatten()


        all_fold_preds.append(fold_preds)

    # ENSEMBLE
    all_fold_preds = torch.stack(all_fold_preds)   # [K, N]
    final_preds = all_fold_preds.mean(dim=0)       # [N]
    final_labels = fold_labels

    rmse = torch.sqrt(((final_preds - final_labels) ** 2).mean())
    corr, p_value = pearsonr(final_preds, final_labels)
    ci = concordance_index(final_labels.numpy(), final_preds.numpy())
    return rmse.item(), corr, ci


if __name__ == "__main__":

    
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args        = parse_args()
    seed_num = int(args.num_seeds)
    set_seed(seed_num)
    df          = pd.read_csv(args.csv_path)
    batch_size   = int(args.batch_size)
    
    # ===== Model =====
    best_val_loss   = float("inf")
    num_epochs      = 100
    mseLoss         = nn.MSELoss()
    folds = ["Train_1", "Train_2", "Train_3", "Train_4", "Train_5"]

    timestamp = datetime.now().strftime(f"%Y%m%d_%H%M%S_dta_{seed_num}")

    log_dir = os.path.join("logs", f"{args.exp_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "train.log")

    def log(msg):
        print(msg)   # print console
        
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    config_path = os.path.join(log_dir, "config.json")

    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    fold_rmses = []
    early_stop_patience = 15
    for val_fold in folds:
        log(f"\n===== Fold: {val_fold} =====")

        model = DTA_Predictor().to(device)   # ✅ reset
        
        # optimizer
        
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=5e-5,
            weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5
        )


        best_val_r2 = float("-inf")

        train_df = df[(df["split"] != val_fold) & (df["split"].str.contains("Train"))]
        val_df   = df[df["split"] == val_fold]

    

        train_dataset = DTA_Dataloader(train_df, args.protein_emb_path, args.fg_instance_path, args.ligand_graph_path)
        val_dataset   = DTA_Dataloader(val_df, args.protein_emb_path, args.fg_instance_path, args.ligand_graph_path)

        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_DTA)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_DTA)
        
        epochs_no_improve = 0
        for epoch in range(num_epochs):

            model.train()
            train_loss = 0

            for batch in tqdm(train_loader):
                
                prot_tensors, prot_masks, batched_graph, fg_indices_tensor, fg_type_tensor, labels = batch

                optimizer.zero_grad()

                prot_tensors        = prot_tensors.to(device)
                prot_masks          = prot_masks.to(device)
                batched_graph       = batched_graph.to(device)
                fg_indices_tensor   = fg_indices_tensor.to(device)
                fg_type_tensor      = fg_type_tensor.to(device)
                labels              = labels.to(device).float()
                preds               = model(prot_tensors, prot_masks, batched_graph, fg_indices_tensor, fg_type_tensor).squeeze(-1)

                
                mse_loss   = mseLoss(preds, labels)
                loss = mse_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / (len(train_loader) + 1e-10)

            # ✅ VALIDATE
            avg_val_loss, rmse, corr, ci, r2 = evaluate(model, val_loader, device, mseLoss)
            print(avg_val_loss, rmse, corr, ci)  
            # scheduler CI
            scheduler.step(ci)
            log(f"Epoch {epoch+1} | Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f} | RMSE {rmse:.4f} | R {corr:.4f} | CI {ci:.4f} | R2 {r2:.4f}")

            # ✅ SAVE BEST
            if best_val_ci < ci:
                best_val_ci   = ci

                save_path       = os.path.join(log_dir, f"best_model_{val_fold}.pth")

                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": avg_val_loss,
                    "rmse": rmse
                }, save_path)
                epochs_no_improve = 0   # ✅ reset counter
                log(f"✅ Saved BEST {val_fold}")
            else:
                epochs_no_improve += 1
            
            # ✅ EARLY STOP
            if epochs_no_improve >= early_stop_patience:
                log(f"⛔ Early stopping triggered at epoch {epoch+1}")
                break

        fold_rmses.append(best_val_loss)
    
    test_df = df[df["split"] == "Test"]
    test_dataset = DTA_Dataloader(
        test_df,
        args.protein_emb_path,
        args.fg_instance_path,
        args.ligand_graph_path
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_DTA
    )
    print(log_dir)
    model_paths = [
        os.path.join(log_dir, f"best_model_{fold}.pth")
        for fold in folds
    ]

    rmse, corr, ci = test_ensemble(model_paths, test_loader, device)
    log(f"\n TEST ENSEMBLE RMSE: | RMSE {rmse:.4f} | R {corr:.4f} | CI {ci:.4f}")
