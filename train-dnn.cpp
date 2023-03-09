#include <iostream>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>

using namespace std;
using namespace dlib;

// A 5x5 conv layer that does 2x downsampling
template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
// A 3x3 conv layer that doesn't do any downsampling
template <long num_filters, typename SUBNET> using con3  = con<num_filters,3,3,1,1,SUBNET>;

// Now we can define the 8x downsampling block in terms of conv5d blocks.  We
// also use relu and batch normalization in the standard way.
template <typename SUBNET> using downsampler  = relu<bn_con<con5d<32, relu<bn_con<con5d<32, relu<bn_con<con5d<32,SUBNET>>>>>>>>>;

// The rest of the network will be 3x3 conv layers with batch normalization and
// relu.  So we define the 3x3 block we will use here.
template <typename SUBNET> using rcon3  = relu<bn_con<con3<32,SUBNET>>>;

// Finally, we define the entire network.   The special input_rgb_image_pyramid
// layer causes the network to operate over a spatial pyramid, making the detector
// scale invariant.  
using net_type  = loss_mmod<con<1,6,6,1,1,rcon3<rcon3<rcon3<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv) try
{
    if (argc != 3)
    {
        cout << "Give the path to the directory and path to training.xml as the argument to this" << endl;
        cout << "   ./train-dnn <path_to_directory> <path_to_training.xml>" << endl;
        cout << endl;
        return 0;
    }
    const std::string faces_directory = argv[1];
    const std::string training_xml = argv[2];

    std::vector<matrix<rgb_pixel>> images_train, images_test;
    std::vector<std::vector<mmod_rect>> face_boxes_train;

    load_image_dataset(images_train, face_boxes_train, training_xml);

    cout << "num training images: " << images_train.size() << endl;

    mmod_options options(face_boxes_train, 40,40);

    cout << "num detector windows: "<< options.detector_windows.size() << endl;
    for (auto& w : options.detector_windows)
        cout << "detector window width by height: " << w.width << " x " << w.height << endl;
    cout << "overlap NMS IOU thresh:             " << options.overlaps_nms.get_iou_thresh() << endl;
    cout << "overlap NMS percent covered thresh: " << options.overlaps_nms.get_percent_covered_thresh() << endl;
 
    net_type net(options);

    net.subnet().layer_details().set_num_filters(options.detector_windows.size());
    dnn_trainer<net_type> trainer(net);
    trainer.set_learning_rate(0.1);
    trainer.be_verbose();
    trainer.set_synchronization_file("mmod_sync", std::chrono::minutes(3));
    trainer.set_iterations_without_progress_threshold(80);

    std::vector<matrix<rgb_pixel>> mini_batch_samples;
    std::vector<std::vector<mmod_rect>> mini_batch_labels; 
    random_cropper cropper;
    cropper.set_chip_dims(200, 200);

    cropper.set_min_object_size(40,40);
    dlib::rand rnd;

    while(trainer.get_learning_rate() >= 1e-4)
    {
        cropper(150, images_train, face_boxes_train, mini_batch_samples, mini_batch_labels);

        for (auto&& img : mini_batch_samples)
            disturb_colors(img, rnd);

        trainer.train_one_step(mini_batch_samples, mini_batch_labels);
    }

    trainer.get_net();
    cout << "done training" << endl;

    net.clean();
    serialize("face_detector.dat") << net;

    cout << trainer << cropper << endl;

    image_window win;
    for (auto&& img : images_train)
    {
        pyramid_up(img);
        auto dets = net(img);
        win.clear_overlay();
        win.set_image(img);
        for (auto&& d : dets)
            win.add_overlay(d);
        cout << "Hit enter to process the next image..." << endl;
        cin.get();
    }
    return 0;

}
catch(std::exception& e)
{
    cout << e.what() << endl;
}




