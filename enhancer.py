import os

import cv2
import numpy as np
import tensorflow as tf


class Enhancer:
    def __init__(self, gpu=True):
        self.gpu = gpu
        self._files = []
        self._model = None
        self._locked = False
        self._sess = None
        self._graph = _Graph()
        self._result = []
        self._available = True
        self.model_available = lambda: self._sess is not None
        if not gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    def _init_graph(self):
        graph = tf.Graph()
        with graph.as_default():
            self._graph.image_ph = tf.placeholder(dtype=tf.float32, shape=(None, None, 3), name="ph")
            self._graph.height = tf.placeholder(dtype=tf.int32)
            self._graph.width = tf.placeholder(dtype=tf.int32)

            input_image = tf.cast(self._graph.image_ph, dtype=tf.float32)
            input_image = input_image / 127.5 - 1

            with tf.gfile.GFile(self._model, 'rb') as model_file:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(model_file.read())

            try:
                self._graph.output_image = tf.import_graph_def(graph_def,
                                                               input_map={'input_image': input_image},
                                                               return_elements=['output_image:0'],
                                                               name='output')
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                self._sess = tf.Session(graph=graph, config=config)

            except Exception as e:
                try:
                    self._sess.close()
                    raise e
                except Exception as e:
                    self._sess = None
                    raise e

    def close(self):
        self._sess.close()
        self._lock()

    def load_model(self, model_path):
        self._model = model_path
        self._init_graph()
        self._unlock()

    def add_files(self, file: dict):  # for batch process
        """
        :param file: {
            'path'          : str,  where's the file.
            'denoise'       : bool, denoise before process.
            'denoise_after' : bool, denoise after process.
        }
        :return:
            self
        """

        self._files.append(file)

    def empty(self):
        self._files = []

    def is_available(self):
        return self._available

    def batch_process(self):
        self._available = False

        for file in self._files:
            path = file['path']
            save_path = file['save_path']
            denoise = file['denoise']
            denoise_after = file['denoise_after']
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            try:
                h, w, _ = image.shape

                if denoise:
                    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 5, 5)

                image = image[h % 4:, w % 4:, :]
                h, w, _ = image.shape

                [[[result_img]]] = self._sess.run([self._graph.output_image], feed_dict={
                    self._graph.image_ph: image,
                    self._graph.height: h,
                    self._graph.width: w
                })
                result_img = np.asarray(result_img)

                if denoise_after:
                    result_img = cv2.fastNlMeansDenoisingColored(result_img, None, 10, 10, 5, 5)

                result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, result_img)

            except Exception as e:
                print("Something went wrong!")
                print(str(e))

        self._available = True

    def sample(self, image, denoise=False, denoise_after=False):
        image = image[:,:,:3]
        self._available = False
        try:
            h, w, _ = image.shape
            if denoise:
                image = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)

            h, w, _ = image.shape

            [[[result_img]]] = self._sess.run([self._graph.output_image], feed_dict={
                self._graph.image_ph: image,
                self._graph.height: h,
                self._graph.width: w
            })
            result_img = np.asarray(result_img)

            if denoise_after:
                result_img = cv2.fastNlMeansDenoisingColored(result_img, None, 10, 10, 5, 5)

            return result_img
        except Exception as e:
            print("Something went wrong!")
            print(str(e))
            return np.zeros_like(image)

        finally:
            self._available = True

    def _lock(self):
        self._available = False

    def _unlock(self):
        self._available = True


class _Graph:
    def __init__(self):
        self.output_image = None
        self.image_ph = None
        self.height = None
        self.width = None


def add_gaussian_noise(image, mean=0, std=0.001):
    """
        添加高斯噪声
        mean : 均值
        var : 方差
    """
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, std ** 0.5, image.shape)
    print(np.mean(noise ** 2) - np.mean(noise) ** 2)
    out = image + noise
    if image.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out
