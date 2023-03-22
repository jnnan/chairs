from torch.utils.data import DataLoader
from ahoi_utils import *
from models_cvae import CVAE
from tqdm import tqdm
from dataloaders import AhoiDataset


if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Hyper-parameters

    hidden_dim = 16
    out_conv_channels = 512
    num_epochs = 100
    batch_size = 12
    n_grid = 64
    learning_rate = 0.0001



    train_dataset = AhoiDataset(data_folder=DATA_FOLDER, n_grid=n_grid, add_human=False, add_contrast=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    
    model = CVAE(dim=n_grid, out_conv_channels=out_conv_channels, hidden_dim=hidden_dim).to(device)

    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 500, 2)

    # Train the model
    total_step = len(train_loader)

    # Start training
    step_id = 0
    for epoch in range(num_epochs):

        for i, (output, idx) in enumerate(tqdm(train_loader)):

            x = output['occ']
            x_fake = output['occ_fake']

            model.train()
            # Forward pass
            x = x.to(device)
            x_fake = x_fake.to(device)

            x_rec, mu, log_var, z = model(x)
            z_fake, _ = model.encoder(x_fake)

            # Compute reconstruction loss and kl divergence
            # For KL divergence, see Appendix B in VAE paper
            reconst_loss = F.binary_cross_entropy(x_rec, x, size_average=False, weight=1 + 10 / (1 + 0.01 * step_id) * x)
            kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            contra_loss = torch.sum(torch.square(z) - torch.square(z_fake), dim=1)
            contra_loss[contra_loss < 0] = 0
            contra_loss = torch.sum(contra_loss)

            if (i + 1) % 100 == 0:
                model.eval()

                with torch.no_grad():
                    x_bin_rec = x_rec.clone()
                    x_bin_rec[x_bin_rec >= 0.5] = 1.
                    x_bin_rec[x_bin_rec < 0.5] = 0.
                    x_bin_rec = x_bin_rec.to(torch.bool)
                    x_bin = x.clone().to(torch.bool)
                    precision = torch.count_nonzero(x_bin[x_bin_rec]) / torch.count_nonzero(x_bin)
                    recall = torch.count_nonzero(x_bin_rec[x_bin]) / torch.count_nonzero(x_bin_rec)

                lr = optimizer.param_groups[0]['lr']
                print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}, Contra Loss: {:.4f} precision: {:.4f}, recall: {:4f}, lr: {:f}"
                    .format(epoch + 1, num_epochs, i + 1, len(train_loader), reconst_loss.item(), kl_div.item(), contra_loss.item(),
                            precision.item(), recall.item(), lr))

            # Backprop and optimize
            model.train()
            loss = reconst_loss + 100. * kl_div + 100. * contra_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            step_id += 1


