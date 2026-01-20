import sys
import os
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
from model.modules import *
from model.loss import *
from dataloader.dataloader import *
import argparse
from tqdm import tqdm

@torch.no_grad()
def evaluate(model, dataloader, regression, alignloss, device):
    model.eval()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Evaluating"):
        prot_vec  = batch['prot_tensors'].to(device)
        prot_mask = batch['prot_mask'].to(device).float()
        ligand_vec = batch['ligand_ctx'].to(device)
        ligand_mask = batch['fg_mask'].to(device).float()
        logits      = batch['logits'].to(device).float()
        values = batch['values'].to(device)

        preds, latent_vector = model(ligand_vec, prot_vec, logits, ligand_mask, prot_mask)
        
        regression_loss  = regression(preds, values)
        loss = regression_loss
        total_loss += loss.item() * values.size(0)

    mse = total_loss / len(dataloader.dataset)
    rmse = mse ** 0.5
    return rmse

def train_epoch(model, dataloader, optimizer, regression, alignloss, lambda_value, device):
    model.train()
    total_loss = 0
    total_rmse = 0
    len_data_loader = len(dataloader)
    for index_batch, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        prot_vec  = batch['prot_tensors'].to(device)
        prot_mask = batch['prot_mask'].to(device).float()
        ligand_vec = batch['ligand_ctx'].to(device)
        ligand_mask = batch['fg_mask'].to(device).float()
        logits      = batch['logits'].to(device).float()
        values = batch['values'].to(device)

        preds, latent_vector = model(ligand_vec, prot_vec, logits, ligand_mask, prot_mask)
        
        regression_loss  = regression(preds, values)
        align_loss       = alignloss(latent_vector, values)
        loss = regression_loss + align_loss*lambda_value
        loss.backward()
        optimizer.step()
        total_rmse += regression_loss.item() * values.size(0)
        total_loss += loss.item() * values.size(0)
        if index_batch%100 == 0:
            print(f'{index_batch}/{len_data_loader} | Loss MSE: {regression_loss.item()}, Loss Align: {(align_loss*lambda_value).item()}')
        
    mse = total_rmse / len(dataloader.dataset)
    rmse = mse ** 0.5
    return rmse


def train_model(model, train_loader, val_loader, regression, alignloss, optimizer, device, num_epochs=30, lambda_value = 5, save_path="best_model.pth"):
    best_val_rmse = float('inf')
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_rmse = train_epoch(model, train_loader, optimizer, regression, alignloss, lambda_value, device)
        val_rmse = evaluate(model, val_loader,  regression, alignloss, device)
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), save_path)
            print("✅ Best model saved!")
        print(f"Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}")
        print(f'Best val: ', best_val_rmse)


@torch.no_grad()
def test_model(model, test_loader, regression, alignloss, device, model_path="best_model.pth"):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    test_rmse = evaluate(model, test_loader, regression, alignloss, device)
    print(f"🎯 Test RMSE: {test_rmse:.4f}")
    return test_rmse


def parse_args():
    parser = argparse.ArgumentParser(description="Binding Affinity Prediction")
    parser.add_argument(
        "--base_path",
        type=str,
        default="evaluation/extracted_features",
        help="Path to extracted feature directory",
    )
    parser.add_argument(
        "--lambda_value",
        type=float,
        default=1.0,
        help="Weight for alignment loss",
    )
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    
    train_dataset = BindingAffinityPrediction_Dataset(args.base_path, 'train')
    val_dataset = BindingAffinityPrediction_Dataset(args.base_path, 'val')
    test_dataset = BindingAffinityPrediction_Dataset(args.base_path, 'test')
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_binding_affinity)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_binding_affinity)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_binding_affinity)
    feature_dim = 960
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Bidirectional_Interaction_Type_Attention(emb_dim = 960, num_types=7, hidden=512, nonneg=True, use_sigmoid=False).to(device)
    # state_dict = torch.load('best_predictor.pt', map_location=device)
    # model.load_state_dict(state_dict)
    regressionloss = nn.MSELoss()
    alignloss = LatentAlignmentLoss(tau=0.1, uniform_weight=0.1).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    train_model(model, train_loader, test_loader, regressionloss, alignloss, optimizer, device, num_epochs=10, lambda_value = args.lambda_value, save_path="checkpoints/best_predictor.pt")
    test_model(model, test_loader, regressionloss, alignloss, device, model_path="checkpoints/best_predictor.pt")

