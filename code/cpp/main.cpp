#include "Viewer.h"

#include <iostream>
#include <unistd.h>

#include <GL/glew.h>
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
    const int gridsize = 3; // per dimension
    const int nbvoxel  = gridsize * gridsize * gridsize;

    // bake kernel
    VectorXd area = VectorXd::Constant(nbvoxel, 1.0);
    //mesh.computeVertArea(area);

    // TODO: need to be defined
    SparseMatrix matrix;
    mesh.buildHeatKernel(matrix, opt.gamma);
    const double mu = double(gridsize) / double(gridsize - 10);
    // mu = N/(N-10);
    // blur = load_filtering('imgaussian', N);
    // K = @(x)blur(x,mu);
    // Kv = @(x)apply_3d_f
    // unc(K,x);


    // apply a gaussian filter in each dimension
    // p is the N*N*N voxelization in one column (i.e size == [N*N*N, 1])
    const auto Kv = [&] (VectorXd& p) VectorXd {
      VectorXd ret = p;

      

      return ret;
    };

    LinearSolver lsolver;
    lsolver.factorizePosDef(matrix);

    // set OT solver
    ot::ConvSolver otsolver(opt, area, lsolver);

    std::vector<VectorXd> p(nb_shape);

    for(int i = 0; i < nb_shape; ++i)
    {
      p[i] = VectorXd::Constant(nbvoxel, 0.0);
      std::cout << p[i];
    }

    p[0](0) = 0.5;
    p[0](nbvoxel-1) = 0.5;

    p[1](0) = 0.5;
    p[1](nbvoxel-1) = 0.5;

    VectorXd out_barycenter;
    VectorXd alpha(nb_shape);
    alpha[0] = 0.5;
    alpha[1] = 0.5;
    otsolver.computeBarycenter(p, alpha, out_barycenter,
                               use_sharp, verbose);

    std::cout << out_barycenter;
    // // gui
    // Viewer viewer;
    // viewer.meshPtr = &mesh;
    // viewer.verbose = verbose;
    // viewer.solverPtr = &otsolver;
    // viewer.clearData();
    // viewer.init(argc,argv);
    return 1;
}
