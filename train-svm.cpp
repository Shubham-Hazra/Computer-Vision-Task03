#include <dlib/svm_threaded.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/data_io.h>

#include <iostream>
#include <fstream>


using namespace std;
using namespace dlib;

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{  

    try
    {
        // In this example we are going to train a face detector based on the
        // small faces dataset in the examples/faces directory.  So the first
        // thing we do is load that dataset.  This means you need to supply the
        // path to this faces folder as a command line argument so we will know
        // where it is.
        if (argc != 3)
        {
            cout << "Give the path to the directory and path to training.xml as the argument to this" << endl;
            cout << "   ./train-svm <path_to_directory> <path_to_training.xml>" << endl;
            cout << endl;
            return 0;
        }
        const std::string faces_directory = argv[1];
        const std::string training_xml = argv[2];
        // The faces directory contains a training dataset and a separate
        // testing dataset.  The training data consists of 4 images, each
        // annotated with rectangles that bound each human face.  The idea is 
        // to use this training data to learn to identify human faces in new
        // images.  
        // 
        // Once you have trained an object detector it is always important to
        // test it on data it wasn't trained on.  Therefore, we will also load
        // a separate testing set of 5 images.  Once we have a face detector
        // created from the training data we will see how well it works by
        // running it on the testing images. 
        // 
        // So here we create the variables that will hold our dataset.
        // images_train will hold the 4 training images and face_boxes_train
        // holds the locations of the faces in the training images.  So for
        // example, the image images_train[0] has the faces given by the
        // rectangles in face_boxes_train[0].
        dlib::array<array2d<unsigned char> > images_train;
        std::vector<std::vector<rectangle> > face_boxes_train;

        // Now we load the data.  These XML files list the images in each
        // dataset and also contain the positions of the face boxes.  Obviously
        // you can use any kind of input format you like so long as you store
        // the data into images_train and face_boxes_train.  But for convenience
        // dlib comes with tools for creating and loading XML image dataset
        // files.  Here you see how to load the data.  To create the XML files
        // you can use the imglab tool which can be found in the tools/imglab
        // folder.  It is a simple graphical tool for labeling objects in images
        // with boxes.  To see how to use it read the tools/imglab/README.txt
        // file.
        load_image_dataset(images_train, face_boxes_train,training_xml);
        // Now we do a little bit of pre-processing.  This is optional but for
        // this training data it improves the results.  The first thing we do is
        // increase the size of the images by a factor of two.  We do this
        // because it will allow us to detect smaller faces than otherwise would
        // be practical (since the faces are all now twice as big).  Note that,
        // in addition to resizing the images, these functions also make the
        // appropriate adjustments to the face boxes so that they still fall on
        // top of the faces after the images are resized.
        upsample_image_dataset<pyramid_down<2> >(images_train, face_boxes_train);
        // Since human faces are generally left-right symmetric we can increase
        // our training dataset by adding mirrored versions of each image back
        // into images_train.  So this next step doubles the size of our
        // training dataset.  Again, this is obviously optional but is useful in
        // many object detection tasks.
        add_image_left_right_flips(images_train, face_boxes_train);
        cout << "num training images: " << images_train.size() << endl;


        // Finally we get to the training code.  dlib contains a number of
        // object detectors.  This typedef tells it that you want to use the one
        // based on Felzenszwalb's version of the Histogram of Oriented
        // Gradients (commonly called HOG) detector.  The 6 means that you want
        // it to use an image pyramid that downsamples the image at a ratio of
        // 5/6.  Recall that HOG detectors work by creating an image pyramid and
        // then running the detector over each pyramid level in a sliding window
        // fashion.   
        typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type; 
        image_scanner_type scanner;
        // The sliding window detector will be 80 pixels wide and 80 pixels tall.
        scanner.set_detection_window_size(80, 80); 
        structural_object_detection_trainer<image_scanner_type> trainer(scanner);
        // Set this to the number of processing cores on your machine.
        trainer.set_num_threads(4);  
        // The trainer is a kind of support vector machine and therefore has the usual SVM
        // C parameter.  In general, a bigger C encourages it to fit the training data
        // better but might lead to overfitting.  You must find the best C value
        // empirically by checking how well the trained detector works on a test set of
        // images you haven't trained on.  Don't just leave the value set at 1.  Try a few
        // different C values and see what works best for your data.
        trainer.set_c(1);
        // We can tell the trainer to print it's progress to the console if we want.  
        trainer.be_verbose();
        // The trainer will run until the "risk gap" is less than 0.01.  Smaller values
        // make the trainer solve the SVM optimization problem more accurately but will
        // take longer to train.  For most problems a value in the range of 0.1 to 0.01 is
        // plenty accurate.  Also, when in verbose mode the risk gap is printed on each
        // iteration so you can see how close it is to finishing the training.  
        trainer.set_epsilon(0.01);


        // Now we run the trainer.  For this example, it should take on the order of 10
        // seconds to train.
        object_detector<image_scanner_type> detector = trainer.train(images_train, face_boxes_train);

        // Now that we have a face detector we can test it.  The first statement tests it
        // on the training data.  It will print the precision, recall, and then average precision.
        cout << "training results: " << test_object_detection_function(detector, images_train, face_boxes_train) << endl;

        // If you have read any papers that use HOG you have probably seen the nice looking
        // "sticks" visualization of a learned HOG detector.  This next line creates a
        // window with such a visualization of our detector.  It should look somewhat like
        // a face.
        image_window hogwin(draw_fhog(detector), "Learned fHOG detector");

        // Like everything in dlib, you can save your detector to disk using the
        // serialize() function.
        serialize("face_detector.svm") << detector;
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------

