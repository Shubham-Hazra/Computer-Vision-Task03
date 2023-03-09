#include <dlib/image_processing.h>
#include <dlib/data_io.h>
#include <iostream>

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

std::vector<std::vector<double> > get_interocular_distances (
    const std::vector<std::vector<full_object_detection> >& objects
);

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    try
    {
        if (argc != 3)
        {
            cout << "Give the path to the directory and path to training.xml as the argument to this" << endl;
            cout << "   ./train-shape <path_to_directory> <path_to_training.xml>" << endl;
            cout << endl;
            return 0;
        }
        const std::string faces_directory = argv[1];
        const std::string training_xml = argv[2];
    
        dlib::array<array2d<unsigned char> > images_train;
        std::vector<std::vector<full_object_detection> > faces_train;

        load_image_dataset(images_train, faces_train, training_xml);

        shape_predictor_trainer trainer;

        trainer.set_oversampling_amount(300);

        trainer.set_nu(0.2);
        trainer.set_tree_depth(4);

        trainer.set_num_threads(4);

        trainer.be_verbose();

        shape_predictor sp = trainer.train(images_train, faces_train);

        serialize("shape_predictor.dat") << sp;
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------

double interocular_distance (
    const full_object_detection& det
)
{
    dlib::vector<double,2> l, r;
    double cnt = 0;

    for (unsigned long i = 36; i <= 41; ++i) 
    {
        l += det.part(i);
        ++cnt;
    }
    l /= cnt;

    cnt = 0;
    for (unsigned long i = 42; i <= 47; ++i) 
    {
        r += det.part(i);
        ++cnt;
    }
    r /= cnt;

    return length(l-r);
}

std::vector<std::vector<double> > get_interocular_distances (
    const std::vector<std::vector<full_object_detection> >& objects
)
{
    std::vector<std::vector<double> > temp(objects.size());
    for (unsigned long i = 0; i < objects.size(); ++i)
    {
        for (unsigned long j = 0; j < objects[i].size(); ++j)
        {
            temp[i].push_back(interocular_distance(objects[i][j]));
        }
    }
    return temp;
}

// ----------------------------------------------------------------------------------------

