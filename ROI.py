from scipy.signal import savgol_filter
from skimage.morphology import disk
from skimage.filters.rank import entropy
import os
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
import numpy as np
import scipy  # to upscale the image
import matplotlib.pyplot as plt
import cv2
import skimage
from keras.applications.resnet import ResNet50, preprocess_input
from keras.models import Model
from PIL import Image
import time
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_ubyte, img_as_float

H2Max = 0
i = 1
# start=time.time()

input_folder = "DataSet"
output_folders = ["GAP(R1)","R1", "Phase", "Entropy", "R2_Mask", "R2", "Mask(R1+R2)", "ROI"]


def ROI(input_folder):
    for folder in output_folders:
        os.makedirs(folder, exist_ok=True)

    # Get the list of image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    # Iterate over each image in file
    for image_file in image_files:
        # Open the image file
        image_path = os.path.join(input_folder, image_file)
        dim = (224, 224)
        img_c1 = cv2.imread(image_path)
        # dim = (224, 224)
        img_c1 = cv2.resize(img_c1, dim, interpolation=cv2.INTER_AREA)
        main = img_c1
        img_c1_gray = cv2.cvtColor(img_c1, cv2.COLOR_BGR2GRAY)

        #####   R1   #####
        img = Image.fromarray(img_c1, 'RGB')
        img = np.array(img)
        img_tensor = np.expand_dims(img, axis=0)
        preprocessed_img = preprocess_input(img_tensor)
        model = ResNet50(weights='imagenet')

        last_layer_weights = model.layers[-1].get_weights()[0]  # Predictions layer

        ResNet_model = Model(inputs=model.input,
                             outputs=(model.layers[-4].output, model.layers[-1].output))

        last_conv_output, pred_vec = ResNet_model.predict(preprocessed_img)

        last_conv_output = np.squeeze(last_conv_output)  # 7x7x2048

        pred = np.argmax(pred_vec)

        h = int(img.shape[0] / last_conv_output.shape[0])
        w = int(img.shape[1] / last_conv_output.shape[1])
        upsampled_last_conv_output = scipy.ndimage.zoom(last_conv_output, (h, w, 1), order=1)
        last_layer_weights_for_pred = last_layer_weights[:, pred]  # dim: (2048,)
        heat_map = np.dot(upsampled_last_conv_output.reshape((224 * 224, 2048)),
                          last_layer_weights_for_pred).reshape(224, 224)  # dim: 224 x 224
        GAP_output_path = os.path.join(output_folders[0], image_file)
        plt.imsave(GAP_output_path, heat_map, cmap='gray')

        image1 = cv2.imread(GAP_output_path)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(image1, 170, 255, cv2.THRESH_BINARY)
        t = skimage.filters.threshold_otsu(thresh1)
        binary_mask = thresh1 > t
        selection = main.copy()
        selection[~binary_mask] = 0
        R1_output_path = os.path.join(output_folders[1], image_file)
        cv2.imwrite(R1_output_path, selection)



        ###   R2  ####

        # CannyEdge Detection
        t_lower = 1  # Lower Threshold
        t_upper = 50  # Upper threshold
        aperture_size = 7  # Aperture size
        L2Gradient = True  # Boolean

        edge = cv2.Canny(img_c1_gray, t_lower, t_upper,
                         apertureSize=aperture_size,
                         L2gradient=L2Gradient)

        def gaussianLP(D0, imgShape):
            base = np.zeros(imgShape[:2])

            rows, cols = imgShape[:2]
            center = (rows / 2, cols / 2)
            for x in range(cols):
                for y in range(rows):
                    base[y, x] = np.exp(((-distance((y, x), center) ** 2) / (2 * (D0 ** 2))))
            return base

        def gaussianHP(D0, imgShape):
            base = np.zeros(imgShape[:2])

            rows, cols = imgShape[:2]
            center = (rows / 2, cols / 2)

            for x in range(cols):
                for y in range(rows):
                    base[y, x] = 1 - np.exp(((-distance((y, x), center) ** 2) / (2 * (D0 ** 2))))
            return base

        def distance(point1, point2):
            return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

        # Fourier Transform
        original = np.fft.fft2(img_c1_gray)
        center = np.fft.fftshift(original)

        # Highpass
        HighPassCenter = center * gaussianHP(50, img_c1_gray.shape)
        HighPass = np.fft.ifftshift(HighPassCenter)
        inverse_HighPass = np.fft.ifft2(HighPass)
        highpass = np.abs(inverse_HighPass)

        # Lowpass
        LowPassCenter = center * gaussianLP(50, img_c1_gray.shape)
        LowPass = np.fft.ifftshift(LowPassCenter)
        inverse_LowPass = np.fft.ifft2(LowPass)
        lowpass = np.abs(inverse_LowPass)

        # Amplitude
        fft = np.fft.fftshift(np.fft.fft2(edge))########   Edge/Highpass/Lowpass
        amplitude = np.sqrt(np.real(fft) ** 2 + np.imag(fft) ** 2)

        # Phase
        fft1 = np.fft.fftshift(np.fft.fft2(img_c1_gray))
        phase = np.arctan2(np.imag(fft1), np.real(fft1))

        # Combining Phase and Amplitude
        comb = np.multiply(amplitude, np.exp(1j * phase))
        comb = np.real(np.fft.ifft2(comb))
        comb = np.abs(comb)
        phase_output_path = os.path.join(output_folders[2], image_file)
        plt.imsave(phase_output_path, comb, cmap='gray')

        ##Local Entropy
        img = cv2.imread(phase_output_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img1 = entropy(img, disk(5))

        entropy_output_path = os.path.join(output_folders[3], image_file)
        plt.imsave(entropy_output_path, img1, cmap='magma')

        # Histogram analysis
        image = skimage.io.imread(fname=entropy_output_path, as_gray=True)

        histogram, bin_edges = np.histogram(image, bins=256, range=(0, 1))
        histgram1 = histogram / (histogram.max())

        # Savitzky Golay

        arr = savgol_filter(histgram1, 15, 1, mode='interp')
        arr1 = []

        # Finds The Peaks of Histogram
        peaks, _ = find_peaks(arr, height=0.5, distance=50)

        ##HMAX2
        index = peaks[len(peaks) - 1]
        # Y(Gamma)
        CutPoint = (arr[index] * 90) / 100

        # For Histogram data visualization
        for j in range(1, 255):
            if j >= index:
                if arr[j] <= CutPoint:
                    arr1.append(arr[j])
                else:
                    arr1.append(0)
                    H2Max = j + 1


            else:
                arr1.append(0)

        # Generating mask image after histogram analysis
        float_img = img_as_float(image)
        sigma_est = np.mean(estimate_sigma(image, multichannel=False))
        denoise_img = denoise_nl_means(float_img, h=1.15 * sigma_est, fast_mode=True,
                                       patch_size=5, patch_distance=3, multichannel=False)
        denoise_img_as_8byte = img_as_ubyte(denoise_img)
        segm1 = (denoise_img_as_8byte >= H2Max) & (denoise_img_as_8byte <= 255)

        all_segments = np.zeros(
            (denoise_img_as_8byte.shape[0], denoise_img_as_8byte.shape[1], 3))  # nothing but denoise img size but blank

        all_segments[segm1] = (1, 1, 1)
        all_segments = gaussian_filter(all_segments, sigma=5)

        # Save mask image of R2
        pik_output_path = os.path.join(output_folders[4], image_file)
        plt.imsave(pik_output_path, all_segments, cmap='gray')

        img = cv2.imread(pik_output_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        t = skimage.filters.threshold_otsu(img)
        binary_mask = img > t
        selection = img_c1.copy()
        selection[~binary_mask] = 0
        amrsegment_output_path = os.path.join(output_folders[5], image_file)
        cv2.imwrite(amrsegment_output_path, selection)

        src1 = cv2.imread(GAP_output_path)
        img = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)
        src2 = cv2.imread(amrsegment_output_path)
        img1 = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)
        ret, thresh2 = cv2.threshold(img1, 1, 255, cv2.THRESH_BINARY)

        # img=cv2.bitwise_and(thresh1, thresh2)

        # # [blend_images]
        beta = 1
        gamma = 0
        alpha = 1
        dst = cv2.addWeighted(thresh1, alpha, thresh2, beta, gamma)
        Final_output_Mask_path = os.path.join(output_folders[6], image_file)
        cv2.imwrite(Final_output_Mask_path, dst)

        ##Generate Final Localized image
        t = skimage.filters.threshold_otsu(dst)
        binary_mask = dst > t
        selection = main.copy()
        selection[~binary_mask] = 0
        Final_output_path = os.path.join(output_folders[7], image_file)
        cv2.imwrite(Final_output_path, selection)




ROI(input_folder)
# end=time.time()
# t=end-start
# print(t)
