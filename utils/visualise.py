import wandb

import matplotlib.pyplot as plt


def visualise_latents(enf, coords, latents, img, img_shape, save_to_disk=True, wandb_log=False):
    # Unpack latents
    (p, c, g) = latents

    # Recon with ENF
    enf_out = enf(coords, p, c, g)

    # Reshape out and img to image format
    out = enf_out[0].reshape(*img_shape).detach().numpy()
    img = img[0].reshape(*img_shape).detach().numpy()

    # Create matplotlib figure
    plt.figure()
    plt.subplot(131)
    plt.imshow(img)
    plt.title("Original")
    
    plt.subplot(132)
    plt.imshow(out)
    plt.title("Reconstructed")

    # Plot the poses
    # plt.subplot(133)
    # plt.imshow(out)
    # plt.title("Poses")

    # # Poses are [-1, 1], map to [0, img_shape]
    # poses_m = (p + 1) / 2 * img_shape[0]
    # plt.scatter(poses_m[0, :, 0], poses_m[0, :, 1], c='r')

    if save_to_disk:
        plt.savefig("./sample.png")

    if wandb_log:
        wandb.log({"reconstructed": plt})

    plt.close('all')
