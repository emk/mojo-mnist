# Download and install MNIST data using Python.

from layout import row_major, Layout
from std.python import Python, PythonObject
from std.python.numpy import from_numpy_array
from std.utils import IndexList


@fieldwise_init
struct MnistDataset:
    """Mnist dataset.

    Just for fun, we parameterize this at compile-time with the number of
    images we load."""

    comptime image_height = 28
    comptime image_width = 28
    comptime feature_count = Self.image_height * Self.image_width

    comptime image_batch_layout = Layout.row_major(
        1,
        Self.image_height,
        Self.image_width
    ).make_shape_unknown[0]()
    """The layout of our image data. For now, this is documentation."""

    var train_image_count: Int
    var train_images: List[UInt8]
    var train_labels: List[UInt8]

    var test_image_count: Int
    var test_images:  List[UInt8]
    var test_labels:  List[UInt8]

    @staticmethod
    def load_data(train_image_count: Int = 60000, test_image_count: Int = 10000) raises -> Self:
        """Download, cache and load MNIST dataset."""

        var mnists = Python.import_module("mnists")
        var mnist = mnists.MNIST(target_dir="./data")

        var np_train_images = mnist.train_images()
        print("train images", np_train_images.shape)
        var np_train_labels = mnist.train_labels()
        print("train labels", np_train_labels.shape)
        var np_test_images = mnist.test_images()
        print("test images", np_test_images.shape)
        var np_test_labels = mnist.test_labels()
        print("test labels", np_test_labels.shape)

        var train_images = _import_np(np_train_images, train_image_count*Self.feature_count)
        var train_labels = _import_np(np_train_labels, train_image_count)
        var test_images = _import_np(np_test_images, test_image_count*Self.feature_count)
        var test_labels = _import_np(np_test_labels, test_image_count)

        return Self(train_image_count, train_images^, train_labels^, test_image_count, test_images^, test_labels^)


def _import_np(np_array: PythonObject, size: Int) raises -> List[UInt8]:
    """Convert a numpy array to a native List.

    Instead of doing a copy like this, we may want to go straight to GPU buffers."""
    var np_truncated = np_array.reshape(-1)[:size]
    var src_span = from_numpy_array[mut=False, DType.uint8](np_truncated)
    var out = List[UInt8](capacity=size)
    out.extend(src_span)
    return out^


def main() raises:
    var _ = MnistDataset.load_data(60000, 10000)
    print("loaded MNIST")
