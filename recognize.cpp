#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>

using namespace dlib;
using namespace std;



// ----------------------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------------------
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

std::vector<matrix<rgb_pixel>> jitter_image(
    const matrix<rgb_pixel>& img
);

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv) try
{
    if (argc != 4)
    {
        cout << "Run this example by invoking it like this: " << endl;
        cout << "   ./dnn_face_recognition_ex <path_to_svm_model> <path_to_ann_model> <path_to_image>" << endl;
        cout << endl;
        return 1;
    }
    const std::string svm_detector = argv[1];
    const std::string ann_model = argv[2];

    typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;
    object_detector<image_scanner_type> detector;
    deserialize(svm_detector) >> detector;
    shape_predictor sp;
    deserialize("shape_predictor_5_face_landmarks.dat") >> sp;

    anet_type net;
    deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

    matrix<rgb_pixel> img;
    load_image(img, argv[3]);

    image_window win(img); 


    std::vector<matrix<rgb_pixel>> faces;
    for (auto face : detector(img))
    {
        auto shape = sp(img, face);
        matrix<rgb_pixel> face_chip;
        extract_image_chip(img, get_face_chip_details(shape,150,0.25), face_chip);
        faces.push_back(move(face_chip));
 
        win.add_overlay(face);
    }

    if (faces.size() == 0)
    {
        cout << "No faces found in image!" << endl;
        return 1;
    }

 
    std::vector<matrix<float,0,1>> face_descriptors = net(faces);

    std::vector<sample_pair> edges;
    for (size_t i = 0; i < face_descriptors.size(); ++i)
    {
        for (size_t j = i; j < face_descriptors.size(); ++j)
        {
            if (length(face_descriptors[i]-face_descriptors[j]) < 0.6)
                edges.push_back(sample_pair(i,j));
        }
    }
    std::vector<unsigned long> labels;
    const auto num_clusters = chinese_whispers(edges, labels);

    cout << "number of people found in the image: "<< num_clusters << endl;


    std::vector<image_window> win_clusters(num_clusters);
    for (size_t cluster_id = 0; cluster_id < num_clusters; ++cluster_id)
    {
        std::vector<matrix<rgb_pixel>> temp;
        for (size_t j = 0; j < labels.size(); ++j)
        {
            if (cluster_id == labels[j])
                temp.push_back(faces[j]);
        }
        win_clusters[cluster_id].set_title("face cluster " + cast_to_string(cluster_id));
        win_clusters[cluster_id].set_image(tile_images(temp));
    }

    cout << "Press 'n' to recognize using ANN" << endl;
    while(true)
    {
        if (cin.get() == 'n')
            break;
    }
    net_type detector_ann;
    deserialize(ann_model) >> detector_ann;  

    image_window win_ann(img); 

    std::vector<matrix<rgb_pixel>> faces_ann;
    for (auto face : detector_ann(img))
    {
        auto shape = sp(img, face);
        matrix<rgb_pixel> face_chip;
        extract_image_chip(img, get_face_chip_details(shape,150,0.25), face_chip);
        faces_ann.push_back(move(face_chip));
        win_ann.add_overlay(face);
    }

    if (faces_ann.size() == 0)
    {
        cout << "No faces found in image!" << endl;
        return 1;
    }
 
    std::vector<matrix<float,0,1>> face_descriptors_ann = net(faces_ann);


    std::vector<sample_pair> edges_ann;
    for (size_t i = 0; i < face_descriptors_ann.size(); ++i)
    {
        for (size_t j = i; j < face_descriptors_ann.size(); ++j)
        {
            if (length(face_descriptors_ann[i]-face_descriptors_ann[j]) < 0.6)
                edges_ann.push_back(sample_pair(i,j));
        }
    }
    std::vector<unsigned long> labels_ann;
    const auto num_clusters_ann = chinese_whispers(edges_ann, labels_ann);

    cout << "number of people found in the image: "<< num_clusters_ann << endl;

    std::vector<image_window> win_clusters_ann(num_clusters_ann);
    for (size_t cluster_id = 0; cluster_id < num_clusters_ann; ++cluster_id)
    {
        std::vector<matrix<rgb_pixel>> temp;
        for (size_t j = 0; j < labels_ann.size(); ++j)
        {
            if (cluster_id == labels_ann[j])
                temp.push_back(faces[j]);
        }
        win_clusters_ann[cluster_id].set_title("face cluster " + cast_to_string(cluster_id));
        win_clusters_ann[cluster_id].set_image(tile_images(temp));
    }

    cout << "Press any key to exit." << endl;
    cin.get();
}
catch (std::exception& e)
{
    cout << e.what() << endl;
}

// ----------------------------------------------------------------------------------------

std::vector<matrix<rgb_pixel>> jitter_image(
    const matrix<rgb_pixel>& img
)
{
    thread_local dlib::rand rnd;

    std::vector<matrix<rgb_pixel>> crops; 
    for (int i = 0; i < 100; ++i)
        crops.push_back(jitter_image(img,rnd));

    return crops;
}

// ----------------------------------------------------------------------------------------

