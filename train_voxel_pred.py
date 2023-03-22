import torch.nn as nn
from torch.utils.data import DataLoader
from dataloaders import AhoiDatasetImage
from models_voxel_pred import VoxelPredNet
from tqdm import tqdm
from ahoi_utils import *


if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters
    input_size = 63
    hidden_size = 500
    output_size = 22
    num_epochs = 100
    batch_size = 12
    learning_rate = 0.0001

    # MNIST dataset
    train_dataset = AhoiDatasetImage(data_folder=DATA_FOLDER, add_human=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    test_dataset = AhoiDatasetImage(data_folder=DATA_FOLDER, add_human=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=6)


    model = VoxelPredNet(hidden_size=16).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 500, 2)


    # Train the model
    total_step = len(train_loader)

    step_id = 0
    for epoch in range(num_epochs):

        for i, (output, idx) in enumerate(tqdm(train_loader)):
            model.train()
            step_id += 1
            img = output['img'].to(device)
            occ_human = output['occ_human'].to(device)
            occ = output['occ'].to(device)
            # bbox = output['pare_bbox']
            img_name = output['img_name']

            # print(bbox, img_name)

            occ_pred = model(img, occ_human)

            reconst_loss = F.mse_loss(occ, occ_pred, size_average=False)

            # Backward and optimize
            optimizer.zero_grad()
            reconst_loss.backward()
            optimizer.step()
            scheduler.step()

            if (i + 1) % 500 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, reconst_loss.item()))


                model.eval()
                with torch.no_grad():
                    reconst_losses = []
                    for i_test, (output_test, idx_test) in enumerate(test_loader):
                        img_test = output_test['img'].to(device)
                        occ_human_test = output_test['occ_human'].to(device)
                        occ_test = output_test['occ'].to(device)

                        occ_pred_test = model(img_test, occ_human_test)
                        reconst_loss = F.mse_loss(occ_test, occ_pred_test, size_average=False)
                        reconst_losses.append(reconst_loss.item())
                        if i_test >= 5:
                            break

                    mean_reconst_loss = np.array(reconst_losses).mean()
                    print(f'TEST LOSS: {mean_reconst_loss}')

