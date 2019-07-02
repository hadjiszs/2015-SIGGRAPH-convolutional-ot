#include "Viewer.h"

#include <iostream>
#include <unistd.h>
#include <fstream>
#include <string>
#include <cmath>
#include <type_traits>

#include "convbarycenter.h"
#include <GL/glew.h>

namespace utils {

  // TODO: express it without memory usage? bench
  const auto gen_gaussian = [] (int sigma, int siz) {
    const int lo = -1 * std::ceil(siz/2);
    const int hi = std::ceil(siz/2);

    VectorXd ret = VectorXd::Constant(hi-lo+1, 1.);

    double sumv= 0.;
    for(int i = 0, v = lo; v <= hi; ++i, ++v) {
      ret(i) = v;
      // std::clog << ret(i) << " ";
      ret(i) = std::exp(-(ret(i)*ret(i)/2.*(sigma*sigma)));
      sumv += ret(i);
    }

    for(int i = 0; i < ret.size(); ++i)
      ret(i) /= sumv;
    // std::clog << std::endl;
    return ret;
  };

  const auto isequal = [] (const VectorXd& lhs, const VectorXd& rhs) {
    const double epsilon = 0.000000001; // OK for Kv comparison
    //const double epsilon = 0.00001; // OK for the gaussian comparison
    bool ret = true;

    assert(lhs.size() == rhs.size() && "not the same size");

    int i = 0;
    for(; ret && i < lhs.size(); ++i) {
      std::clog << "#" << i << " "
                << lhs(i) << " ==? " << rhs(i) << std::endl;
      ret = std::abs(lhs(i) - rhs(i)) < epsilon;
    }

    if(not ret)
      std::cerr << "Error in the iteration #" << i << std::endl;

    return ret;
  };

  const auto readcsv = [] (const std::string& fn) {
    std::ifstream input(fn);
    std::clog << "\n\n" << fn << "\n\n" << std::endl;

    const int nblines = std::count(std::istreambuf_iterator<char>(input),
                                   std::istreambuf_iterator<char>(), '\n');

    input.seekg(0);
    int i = 0;
    VectorXd ret = VectorXd::Constant(nblines, 1.);
    for(std::string line; getline( input, line ); ++i)
    {
      ret(i) = std::stod(line);
    }

    return ret;
  };

}

bool parseCmdLine(int argc, char** argv,
                  ot::Options& opt,
                  char* & meshFile,
                  double& scale,
                  int   & verbose)
{
    int arg;
    while ((arg = getopt(argc, argv, "t:i:g:u:a:e:m:v:cf:d:s:")) != -1)
    {
        switch (arg)
        {
            // Options
            case 't':
                opt.tolerance = atof(optarg);
                break;
            case 'i':
                opt.maxIters = atoi(optarg);
                break;
            case 'g':
                opt.gamma = atof(optarg);
                break;
            case 'u':
                opt.upperEntropy = atof(optarg);
                break;
            case 'e':
                opt.epsilon = atof(optarg);
                break;
            case 'd':
                opt.diffIters = atoi(optarg);
                break;
            // Mesh file
            case 'm':
                meshFile = optarg;
                break;
            // Extra parameters
            case 's':
                scale = atof(optarg);
                break;
            case 'v':
                verbose = atoi(optarg);
                break;
        }
    }

    if (meshFile == 0)
        std::cout << "Usage: " << argv[0]
        << " -m mesh.obj"
        << " [-t tolerance=1e-6]"
        << " [-i maxIters=1000]"
        << " [-d diffusionIter=10]"
        << " [-g gamma=0]"
        << " [-e epsilon=1e-20]"
        << " [-u upperEntropy=1]"
        << " [-s scale=1]"
        << " [-v verbose=0]"
        << std::endl;

    return meshFile;
}

using namespace utils;
using namespace Eigen;

#define N 40 // size of the grid

int main(int argc, char** argv)
{
    ot::Options opt;
    int    verbose  = 0;
    char*  meshFile = 0;
    double scale    = 1.0;
    if (!parseCmdLine(argc, argv, opt, meshFile, scale, verbose)) return 0;

    // load mesh
    TriMesh mesh;
    mesh.read(meshFile);
    mesh.normalize(); // make Area = 1
    std::cout << "MeshArea = " << mesh.computeTotalArea() << std::endl;

    // timestep proportinal to mesh size
    // double h = mesh.computeMaxEdgeLength();
    const double h = 0.2;//mesh.computeMaxEdgeLength();
    std::cout << "h = " << h << std::endl;

    // set gamma a la [Crane et al. 2013]
    if (opt.gamma == 0.) opt.gamma = scale*h*h;
    std::cout << "gamma = " << opt.gamma << std::endl;

    const bool use_sharp = false;
    // const int verbose = 1;

    const int nb_shape = 2;
    const int gridsize = 40; // per dimension
    const int nbvoxel  = gridsize * gridsize * gridsize;

    // bake kernel
    VectorXd area = VectorXd::Constant(nbvoxel, 1.0);

    // const double N = 40.; // == gridsize
    const double mu = N/40.;

    VectorXd p_dummy = readcsv("pdummy.csv");
    VectorXd afterkv = readcsv("afterkv.csv");
    VectorXd H = gen_gaussian(mu, mu*50.); //readcsv("hkernel.csv");
    std::clog << H.size() << "\n" << H << std::endl;
    // apply a gaussian filter in each dimension
    // p is the N*N*N voxelization in one column (i.e size == [N*N*N, 1])

    GridConv<N> grid;

    // assert(isequal(H, gen_gaussian(mu, mu*50.))
    //        && "generating gaussian is not ok");
    assert(isequal(afterkv, grid.Kv(p_dummy, H))
           && "Kv is not correctly implemented");

    return 0;
}
