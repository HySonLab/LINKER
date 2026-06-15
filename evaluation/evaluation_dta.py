import sys

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
from sklearn.metrics import r2_score
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.modules_kiba import *
from model.loss import *
from dataloader.dta_dataloader import *
import argparse
from datetime import datetime
from tqdm import tqdm
from scipy.stats import pearsonr
from lifelines.utils import concordance_index

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


def set_seed(seed=0):
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
    return avg_loss, rmse.item(), corr[0], ci

def test(path, loader, device):
    model = DTA_Predictor().to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
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
    r2 = r2_score(
        all_preds.numpy(),
        all_labels.numpy()
    )
    return rmse.item(), corr, ci, r2

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
        
    r2 = r2_score(
        final_labels.numpy(),
        final_preds.numpy()
    )

    return rmse.item(), corr, ci, r2




if __name__ == "__main__":

    
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args        = parse_args()
    seed_num = int(args.num_seeds)
    set_seed(seed_num)
    df          = pd.read_csv(args.csv_path)
    batch_size   = int(args.batch_size)
    checkpoint_dir = f'checkpoints/davis_dta_0'
    # ===== Model =====
    best_val_loss   = float("inf")
    mseLoss         = nn.MSELoss()
    rankingLoss     = RankingLoss()
    folds = ["Train_1", "Train_2", "Train_3", "Train_4", "Train_5"]

    os.makedirs('evaluation', exist_ok=True)

    log_file = os.path.join(f'{checkpoint_dir}', f"test.log")

    def log(msg):
        print(msg)   # print console
        
        with open(log_file, "a") as f:
            f.write(msg + "\n")
    
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
    print(log_file)
    model_paths = [
        os.path.join(checkpoint_dir, f"best_model_{fold}.pth")
        for fold in folds
    ]
    for idx, path in enumerate(model_paths):
        rmse, corr, ci, r2 = test(path, test_loader, device)
        log(f"\nTEST FOLD {idx} RMSE: | RMSE {rmse:.4f} | R {corr:.4f} | CI {ci:.4f} | R^2 {r2:.4f}")
    rmse, corr, ci, r2 = test_ensemble(model_paths, test_loader, device)
    log(f"\nTEST ENSEMBLE RMSE: | RMSE {rmse:.4f} | R {corr:.4f} | CI {ci:.4f} | R^2 {r2:.4f}")
