# coding: utf-8
"""""
The following Image Morpher script is an updated version of Face Morpher obtained from Alyssa Quek. 
Changes include the removal of Stasm library automatic face point detection, since the feeded images do not contain 
faces, but audio spectrograms. Image points for spectrograms are manually defined in the following adapted version.

Original code:
*    Title: Morph faces with Python OpenCV, Numpy, Scipy
*    Author: Alyssa Quek
*    Date: July 2018
*    Code version: 1.3
*    Availability: https://github.com/alyssaq/face_morpher
*
"""""

# Import external dependencies
from automl_server.settings import AUTO_ML_DATA_PATH

try:
    import os
    import os.path
    import glob
    import math
    import numpy as np
    import sys

    sys.path.insert(0, '../doc')

    import cv2
    from PIL import Image
    import librosa
    import warnings

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import scipy.sparse
    import scipy.spatial as spatial
    from scipy.ndimage import imread
except:
    print("Error: Dependencies are not correctly imported!")


# Align images by resizing, centering and cropping to given size
class Aligner:

    def positive_cap(self, num):
        """ Cap a number to ensure positivity

        :param num: positive or negative number
        :returns: (overflow, capped_number)
        """
        if num < 0:
            return 0, abs(num)
        else:
            return num, 0

    def resize_align(self, img, points, size):
        """ Resize image and associated points, align image to the center
          and crop to the desired size

        :param img: image to be resized
        :param points: *m* x 2 array of points
        :param size: (height, width) tuple of new desired size
        """

        return img, points

    def roi_coordinates(self, rect, size, scale):
        """ Align the rectangle into the center and return the top-left coordinates
        within the new size. If rect is smaller, we add borders.

        :param rect: (x, y, w, h) bounding rectangle of the image
        :param size: (width, height) are the desired dimensions
        :param scale: scaling factor of the rectangle to be resized
        :returns: 4 numbers. Top-left coordinates of the aligned ROI.
          (x, y, border_x, border_y). All values are > 0.
        """
        rectx, recty, rectw, recth = rect
        new_height, new_width = size
        mid_x = int((rectx + rectw / 2) * scale)
        mid_y = int((recty + recth / 2) * scale)
        roi_x = mid_x - int(new_width / 2)
        roi_y = mid_y - int(new_height / 2)

        roi_x, border_x = self.positive_cap(roi_x)
        roi_y, border_y = self.positive_cap(roi_y)

        return roi_x, roi_y, border_x, border_y

    def scaling_factor(self, rect, size):
        """ Calculate the scaling factor for the current image to be
            resized to the new dimensions

        :param rect: (x, y, w, h) bounding rectangle of the image
        :param size: (width, height) are the desired dimensions
        :returns: floating point scaling factor
        """
        new_height, new_width = size
        rect_h, rect_w = rect[2:]

        height_ratio = rect_h / new_height
        width_ratio = rect_w / new_width
        scale = 1
        if height_ratio > width_ratio:
            new_recth = 0.8 * new_height
            scale = new_recth / rect_h
        else:
            new_rectw = 0.8 * new_width
            scale = new_rectw / rect_w

        return scale

    def resize_image(self, img, scale):
        """ Resize image with the provided scaling factor

        :param img: image to be resized
        :param scale: scaling factor for resizing the image
        """
        cur_height, cur_width = img.shape[:2]
        new_scaled_height = int(scale * cur_height)
        new_scaled_width = int(scale * cur_width)

        return cv2.resize(img, (new_scaled_width, new_scaled_height))


# Optional blending of warped image
class Blender:

    def mask_from_points(self, size, points):
        radius = 10  # kernel size
        kernel = np.ones((radius, radius), np.uint8)

        mask = np.zeros(size, np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
        mask = cv2.erode(mask, kernel)

        return mask

    def apply_mask(self, img, mask):
        """ Apply mask to supplied image
        :param img: max 3 channel image
        :param mask: [0-255] values in mask
        :returns: new image with mask applied
        """
        masked_img = np.copy(img)
        num_channels = 3
        for c in range(num_channels):
            masked_img[..., c] = img[..., c] * (mask / 255)

        return masked_img

    def weighted_average(self, img1, img2, percent=0.5):
        if percent <= 0:
            return img2
        elif percent >= 1:
            return img1
        else:
            return cv2.addWeighted(img1, percent, img2, 1 - percent, 0)

    def alpha_feathering(self, src_img, dest_img, img_mask, blur_radius=15):
        mask = cv2.blur(img_mask, (blur_radius, blur_radius))
        mask = mask / 255.0

        result_img = np.empty(src_img.shape, np.uint8)
        for i in range(3):
            result_img[..., i] = src_img[..., i] * mask + dest_img[..., i] * (1 - mask)

        return result_img

    def poisson_blend(self, img_source, dest_img, img_mask, offset=(0, 0)):
        # http://opencv.jp/opencv2-x-samples/poisson-blending
        img_target = np.copy(dest_img)
        import pyamg
        # compute regions to be blended
        region_source = (
            max(-offset[0], 0),
            max(-offset[1], 0),
            min(img_target.shape[0] - offset[0], img_source.shape[0]),
            min(img_target.shape[1] - offset[1], img_source.shape[1]))
        region_target = (
            max(offset[0], 0),
            max(offset[1], 0),
            min(img_target.shape[0], img_source.shape[0] + offset[0]),
            min(img_target.shape[1], img_source.shape[1] + offset[1]))
        region_size = (region_source[2] - region_source[0],
                       region_source[3] - region_source[1])

        # clip and normalize mask image
        img_mask = img_mask[region_source[0]:region_source[2],
                   region_source[1]:region_source[3]]

        # create coefficient matrix
        coff_mat = scipy.sparse.identity(np.prod(region_size), format='lil')
        for y in range(region_size[0]):
            for x in range(region_size[1]):
                if img_mask[y, x]:
                    index = x + y * region_size[1]
                    coff_mat[index, index] = 4
                    if index + 1 < np.prod(region_size):
                        coff_mat[index, index + 1] = -1
                    if index - 1 >= 0:
                        coff_mat[index, index - 1] = -1
                    if index + region_size[1] < np.prod(region_size):
                        coff_mat[index, index + region_size[1]] = -1
                    if index - region_size[1] >= 0:
                        coff_mat[index, index - region_size[1]] = -1
        coff_mat = coff_mat.tocsr()

        # create poisson matrix for b
        poisson_mat = pyamg.gallery.poisson(img_mask.shape)
        # for each layer (ex. RGB)
        for num_layer in range(img_target.shape[2]):
            # get subimages
            t = img_target[region_target[0]:region_target[2],
                region_target[1]:region_target[3], num_layer]
            s = img_source[region_source[0]:region_source[2],
                region_source[1]:region_source[3], num_layer]
            t = t.flatten()
            s = s.flatten()

            # create b
            b = poisson_mat * s
            for y in range(region_size[0]):
                for x in range(region_size[1]):
                    if not img_mask[y, x]:
                        index = x + y * region_size[1]
                        b[index] = t[index]

            # solve Ax = b
            x = pyamg.solve(coff_mat, b, verb=False, tol=1e-10)

            # assign x to target image
            x = np.reshape(x, region_size)
            x[x > 255] = 255
            x[x < 0] = 0
            x = np.array(x, img_target.dtype)
            img_target[region_target[0]:region_target[2],
            region_target[1]:region_target[3], num_layer] = x

        return img_target


# Given 2 images and its image points, warp one image to the other
class Wraper:

    def bilinear_interpolate(self, img, coords):
        """ Interpolates over every image channel
        http://en.wikipedia.org/wiki/Bilinear_interpolation

        :param img: max 3 channel image
        :param coords: 2 x _m_ array. 1st row = xcoords, 2nd row = ycoords
        :returns: array of interpolated pixels with same shape as coords
        """
        int_coords = np.int32(coords)
        x0, y0 = int_coords
        dx, dy = coords - int_coords

        # 4 Neighour pixels
        q11 = img[y0, x0]
        q21 = img[y0, x0 + 1]
        q12 = img[y0 + 1, x0]
        q22 = img[y0 + 1, x0 + 1]

        btm = q21.T * dx + q11.T * (1 - dx)
        top = q22.T * dx + q12.T * (1 - dx)
        inter_pixel = top * dy + btm * (1 - dy)

        return inter_pixel.T

    def grid_coordinates(self, points):
        """ x,y grid coordinates within the ROI of supplied points

        :param points: points to generate grid coordinates
        :returns: array of (x, y) coordinates
        """
        xmin = np.min(points[:, 0])
        xmax = np.max(points[:, 0]) + 1
        ymin = np.min(points[:, 1])
        ymax = np.max(points[:, 1]) + 1
        return np.asarray([(x, y) for y in range(ymin, ymax)
                           for x in range(xmin, xmax)], np.uint32)

    def process_warp(self, src_img, result_img, tri_affines, dst_points, delaunay):
        """
        Warp each triangle from the src_image only within the
        ROI of the destination image (points in dst_points).
        """
        roi_coords = self.grid_coordinates(dst_points)

        # indices to vertices. -1 if pixel is not in any triangle
        roi_tri_indices = delaunay.find_simplex(roi_coords)

        for simplex_index in range(len(delaunay.simplices)):
            coords = roi_coords[roi_tri_indices == simplex_index]
            num_coords = len(coords)
            out_coords = np.dot(tri_affines[simplex_index],
                                np.vstack((coords.T, np.ones(num_coords))))
            x, y = coords.T
            result_img[y, x] = self.bilinear_interpolate(src_img, out_coords)

        return None

    def triangular_affine_matrices(self, vertices, src_points, dest_points):
        """
        Calculate the affine transformation matrix for each
        triangle (x,y) vertex from dest_points to src_points

        :param vertices: array of triplet indices to corners of triangle
        :param src_points: array of [x, y] points to landmarks for source image
        :param dest_points: array of [x, y] points to landmarks for destination image
        :returns: 2 x 3 affine matrix transformation for a triangle
        """
        ones = [1, 1, 1]
        for tri_indices in vertices:
            src_tri = np.vstack((src_points[tri_indices, :].T, ones))
            dst_tri = np.vstack((dest_points[tri_indices, :].T, ones))
            mat = np.dot(src_tri, np.linalg.inv(dst_tri))[:2, :]
            yield mat

    def wrap_image(self, src_img, src_points, dest_points, dest_shape, dtype=np.uint8):
        # Resultant image will not have an alpha channel
        num_chans = 3
        src_img = src_img[:, :, :3]

        rows, cols = dest_shape[:2]
        result_img = np.zeros((rows, cols, num_chans), dtype)

        delaunay = spatial.Delaunay(dest_points)
        tri_affines = np.asarray(list(self.triangular_affine_matrices(
            delaunay.simplices, src_points, dest_points)))

        self.process_warp(src_img, result_img, tri_affines, dest_points, delaunay)

        return result_img


def bgr2rgb(img):
    # OpenCV's BGR to RGB
    rgb = np.copy(img)
    rgb[..., 0], rgb[..., 2] = img[..., 2], img[..., 0]
    return rgb


def check_do_plot(func):
    def inner(self, *args, **kwargs):
        if self.do_plot:
            func(self, *args, **kwargs)

    return inner


def check_do_save(func):
    def inner(self, *args, **kwargs):
        if self.do_save:
            func(self, *args, **kwargs)

    return inner


class Plotter(object):

    def __init__(self, plot=True, rows=0, cols=0, num_images=0, out_folder=None, out_filename=None):
        num_saved_images = len(glob.glob(os.path.join(AUTO_ML_DATA_PATH+'/png_out', "*.png")))
        self.save_counter = num_saved_images + 1
        self.plot_counter = 1
        self.do_plot = plot
        self.do_save = out_filename is not None
        self.out_filename = out_filename
        self.set_filepath(out_folder)

        if (rows + cols) == 0 and num_images > 0:
            # Auto-calculate the number of rows and cols for the figure
            self.rows = np.ceil(np.sqrt(num_images / 2.0))
            self.cols = np.ceil(num_images / self.rows)
        else:
            self.rows = rows
            self.cols = cols

    def set_filepath(self, folder):
        if folder is None:
            self.filepath = None
            return

        if not os.path.exists(folder):
            os.makedirs(folder)
        self.filepath = os.path.join(folder, 'DuraMax Recording #1-{0:03d}.png')
        self.do_save = True

    @check_do_save
    def save(self, img, filename=None):
        if self.filepath:
            filename = self.filepath.format(self.save_counter)
            self.save_counter += 1
        elif filename is None:
            filename = self.out_filename

        img = cv2.resize(img, (128, 128))
        mpimg.imsave(filename, bgr2rgb(img))
        print(filename + ' saved')

    @check_do_plot
    def plot_one(self, img):
        p = plt.subplot(self.rows, self.cols, self.plot_counter)
        p.axes.get_xaxis().set_visible(False)
        p.axes.get_yaxis().set_visible(False)
        plt.imshow(bgr2rgb(img))
        self.plot_counter += 1

    @check_do_plot
    def show(self):
        plt.gcf().subplots_adjust(hspace=0.05, wspace=0,
                                  left=0, bottom=0, right=1, top=0.98)
        plt.axis('off')
        plt.show()

    @check_do_plot
    def plot_mesh(self, points, tri, color='k'):
        """ plot triangles """
        for tri_indices in tri.simplices:
            t_ext = [tri_indices[0], tri_indices[1], tri_indices[2], tri_indices[0]]
            plt.plot(points[t_ext, 0], points[t_ext, 1], color)


def check_write_video(func):
    def inner(self, *args, **kwargs):
        if self.video:
            return func(self, *args, **kwargs)
        else:
            pass

    return inner


# Create a video file with the image frames
class Video(object):
    def __init__(self, filename, fps, w, h):
        self.filename = filename

        if filename is None:
            self.video = None
        else:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.video = cv2.VideoWriter(filename, fourcc, fps, (w, h), True)

    @check_write_video
    def write(self, img, num_times=1):
        for i in range(num_times):
            self.video.write(img[..., :3])

    @check_write_video
    def end(self):
        print(self.filename + ' saved')
        self.video.release()


# Morph between 2 or more images
class Morpher:
    add_boundary_points = False

    blender = Blender()
    wraper = Wraper()

    def boundary_points(self, points):
        """ Produce additional boundary points

        :param points: *m* x 2 np array of x,y points
        :returns: 2 additional points at the top corners
        """
        x, y, w, h = cv2.boundingRect(points)
        buffer_percent = 0.1
        spacerw = int(w * buffer_percent)
        spacerh = int(h * buffer_percent)
        return [[x + spacerw, y + spacerh],
                [x + w - spacerw, y + spacerh]]

    def weighted_average_points(self, start_points, end_points, percent=0.5):
        """ Weighted average of two sets of supplied points

        :param start_points: *m* x 2 array of start image points.
        :param end_points: *m* x 2 array of end image points.
        :param percent: [0, 1] percentage weight on start_points
        :returns: *m* x 2 array of weighted average points
        """
        if percent <= 0:
            return end_points
        elif percent >= 1:
            return start_points
        else:
            return np.asarray(start_points * percent + end_points * (1 - percent), np.int32)

    # Image points are set manually
    def load_image_points(self, path, size):
        img = cv2.imread(path)
        # Manually defined image points, pointing the corners of the image
        points = [(0, 0), (0, 1600), (2400, 0), (2400, 1600)]
        points = np.asarray(points)

        if self.add_boundary_points:
            points = np.vstack([points, self.boundary_points(points)])

        if len(points) == 0:
            print('No image points were provided %s' % path)
            return None, None
        else:
            aligner = Aligner()
            return aligner.resize_align(img, points, size)

    def load_valid_image_points(self, imgpaths, size):
        for path in imgpaths:
            img, points = self.load_image_points(path, size)
            if img is not None:
                print(path)
                yield (img, points)

    def list_imgpaths(self):
        for fname in glob.iglob(AUTO_ML_DATA_PATH + '/png/**/*.png'):
                yield fname

    def alpha_image(self, img, points):

        mask = self.blender.mask_from_points(img.shape[:2], points)
        return np.dstack((img, mask))

    def morph(self, src_img, src_points, dest_img, dest_points, video,
              width=500, height=600, num_frames=20, fps=10,
              out_frames=None, out_video=None, alpha=False, plot=False):
        """
        Create a morph sequence from source to destination image

        :param src_img: ndarray source image
        :param src_img: source image array of x,y image points
        :param dest_img: ndarray destination image
        :param dest_img: destination image array of x,y image points
        :param video: imagemorpher.videoer.Video object
        """
        size = (height, width)
        stall_frames = np.clip(int(fps * 0.15), 1, fps)  # Show first & last longer
        plt = Plotter(plot, num_images=num_frames, out_folder=out_frames)
        num_frames -= (stall_frames * 2)  # No need to process src and dest image

        plt.plot_one(src_img)
        video.write(src_img, 1)

        # Produce morph frames!
        for percent in np.linspace(1, 0, num=num_frames):
            points = self.weighted_average_points(src_points, dest_points, percent)
            #src_image = self.wraper.wrap_image(src_img, src_points, points, size)
            #end_image = self.wraper.wrap_image(dest_img, dest_points, points, size)

            src_image = src_img
            end_image = dest_img
            average_image = self.blender.weighted_average(src_image, end_image, percent)
            average_image = self.alpha_image(average_image, points) if alpha else average_image

            plt.plot_one(average_image)
            plt.save(average_image)
            video.write(average_image)

        plt.plot_one(dest_img)
        video.write(dest_img, stall_frames)
        plt.show()

    # Read the created images and save them as a numpy array
    def generate_image_array(self):
        filenames = glob.glob(os.path.join(AUTO_ML_DATA_PATH+'/png', "*.png"))
        #filenames.sort(key=os.path.getmtime)

        loaded_features = np.array([cv2.imread(fn) for fn in filenames])


        loaded_features = loaded_features / 255.0

        feature_file = os.path.join(AUTO_ML_DATA_PATH+'/npy', 'generated_images.npy')
        print("\nGenerated image array shape: " + str(loaded_features.shape) + "\n")

        for i in range(len(loaded_features)):
            # first order difference, computed over 9-step window
            loaded_features[i, :, :, 1] = librosa.feature.delta(loaded_features[i, :, :, 0])

            # for using 3 dimensional array to use ResNet and other frameworks
            loaded_features[i, :, :, 2] = librosa.feature.delta(loaded_features[i, :, :, 1])

        loaded_features = np.transpose(loaded_features, (0, 2, 1, 3))

        # Save the features file
        np.save(feature_file, loaded_features)

        return loaded_features

    def initalize(self, imgpaths, width=600, height=500, num_frames=20, fps=10,
                  out_frames=None, out_video=None, alpha=False, plot=False):
        """
        Create a morph sequence from multiple images in imgpaths

        :param imgpaths: array or generator of image paths

        """
        video = Video(out_video, fps, width, height)
        images_points_gen = self.load_valid_image_points(imgpaths, (height, width))
        src_img, src_points = next(images_points_gen)
        for dest_img, dest_points in images_points_gen:
            self.morph(src_img, src_points, dest_img, dest_points, video,
                       width, height, num_frames, fps, out_frames, out_video, alpha, plot)
            src_img, src_points = dest_img, dest_points

        video.end()

        self.generate_image_array()

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
num_images = 500  # The total number of images to be generated

# First process: produce more images with lower fps
def morph_img():
    morpher = Morpher()
    morpher.initalize(imgpaths=morpher.list_imgpaths(),
                  width=128, height=128,
                  num_frames=math.floor(num_images*0.7), fps=1,
                  out_frames=AUTO_ML_DATA_PATH+'/png_out', out_video=None,
                  alpha=True, plot=False)

# Resize the destination image
# img = cv2.resize(cv2.imread('/png_out/' + "/dest.png"), (128, 128))