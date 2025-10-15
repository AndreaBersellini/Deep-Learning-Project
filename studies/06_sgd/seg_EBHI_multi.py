from utils_package import *
from utils_package.loaders import build_loaders_aug
from utils_package.net_UNET import Unet
from utils_package.constants import IN_CHANNELS, OUT_CHANNELS, LEARNING_RATE, MIN_EPOCHS, MAX_EPOCHS, DEVICE, EARLY_STOP_COUNT, CLASS_NAMES
from utils_package.train_function import train_loop, test_loop
from utils_package.utils import early_stopping

sys.stdout = open('plots/console_out.txt', 'w', encoding='utf-8')

for i in range(3):
    model = Unet(IN_CHANNELS, OUT_CHANNELS).to(device = DEVICE)
    loss_fn = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr = LEARNING_RATE, momentum=0.99)
    train_loader, test_loader = build_loaders_aug()

    test_losses = []
    stop_count = 0

    print(f"Training experiment {i}")

    start_train = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        train_loop(train_loader, model, optimizer, loss_fn, epoch, hpc=True)
        avg_loss, avg_iou, avg_dice, avg_pxl_err = test_loop(test_loader, model, loss_fn, hpc=True)

        test_losses.append(avg_loss)

        if early_stopping(test_losses): stop_count += 1
        else: stop_count = 0

        if stop_count >= EARLY_STOP_COUNT and epoch >= MIN_EPOCHS:
            break
    
    finish_train = time.perf_counter()

    print(f"Training time: {round(finish_train - start_train, 2)}s")

    print("\nFinal average IoU:")
    for cls, iou in enumerate(avg_iou):
        print(f"{CLASS_NAMES[cls]}: {iou}")
    print("\nFinal average Dice Score:")
    for cls, dice in enumerate(avg_dice):
        print(f"{CLASS_NAMES[cls]}: {dice}")
    print("\nFinal average Pixel Error:")
    for cls, err in enumerate(avg_pxl_err):
        print(f"{CLASS_NAMES[cls]}: {err}")
    print("-" * 50,"\n","-" * 50)

sys.stdout.close()