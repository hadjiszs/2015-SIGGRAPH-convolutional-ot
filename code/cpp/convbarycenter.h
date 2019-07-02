#pragma once

#include <Eigen/Core>

#define NBSHAPE 4

template<typename T>
struct less {
  bool operator()(const T& lhs, const T& rhs) const { return lhs < rhs; };
};

template<typename T, class Compare>
const T& clamp(T&& v, T&& lo, T&& hi, Compare comp) {
  return assert(!comp(hi, lo)),
    comp(v, lo)? lo : comp(hi, v) ? hi : v;
}

template<typename T>
const T& clamp(T&& v, T&& lo, T&& hi) {
  return clamp(v, lo, hi, less<T>());
}

inline
void clamp(VectorXd& q, double zero)
{
    for (int i = 0; i < int(q.size()); ++i) 
    {
        if (q[i] > zero or q[i] < -zero) continue;
        if (q[i] >= 0.0) q[i] =  zero;
        else             q[i] = -zero;
    }
}

inline
double dot_product(const VectorXd& area, const VectorXd& x, const VectorXd& y)
{
    VectorXd tmp = area.array() * y.array();
    return x.dot(tmp);
}

template<int N>
struct GridConv {
  int _width  = N;
  int _depth  = N;
  int _height = N;
  int _niter  = 100;

  VectorXd _H;
  VectorXd _area;

  GridConv() = delete;
  GridConv(const VectorXd h, const VectorXd area)
    : _H(h), _area(area) { }

  double& get(VectorXd& p, int i, int j, int k)
  {
    // array3D[i][j][k]
    // int depth = n, width = n;
    return p[i*(_depth*_width)+j*_depth+k];
  };

  double dotProduct(const VectorXd& x, const VectorXd& y) {
    VectorXd tmp = _area.array() * y.array();
    return x.dot(tmp);
  }

  VectorXd imfilter(VectorXd I, const VectorXd& h, int dim) {
    int w = _width;
    VectorXd vres = I;
    // sort of 1d convolution

    const auto idx = [&] (int i) {
      int lo=0, hi=w-1;
      return clamp(i, lo, hi);
    };

    int DIM = h.size()/2;
    // std::clog << "DIM: " << DIM << std::endl;
    for (int k = 0; k < _depth; k++) {
      for (int j = 0; j < _height; j++) {
        for (int i = 0; i < _width; i++) {
          double res = 0.;
          //std::clog << "[";
          for(int p = 0; p < h.size(); ++p) {
            //std::clog<<h(p)<< "*"<<get(test, k, j, idx(i+p-DIM), ntest)<<"+";
            //res += h(p) * get(test, k, j, idx(i+p-DIM), ntest);
            //res += h(p) * get(test, k, idx(j+p-DIM), i, ntest);
            if(dim==1)
              res += h(p) * get(I, k, j, idx(i+p-DIM));
            else if(dim==2)
              res += h(p) * get(I, k, idx(j+p-DIM), i);
            else
              res += h(p) * get(I, idx(k+p-DIM), j, i);
          }

          get(vres, k, j, i) = res;
          // std::clog << res << " ";
        }
        // std::clog << std::endl;
      }
      // std::clog << "\n next layer " << std::endl;
    }
    return vres;
  }

  VectorXd Kv(VectorXd& p, VectorXd& H) {
    VectorXd ret = p;
    return imfilter(imfilter(imfilter(p, H, 1), H, 2), H, 3);
  };

  VectorXd Kv(VectorXd& p) {
    return Kv(p, _H);
  };

  VectorXd convoWassersteinBarycenter(std::array<VectorXd, NBSHAPE>& p,
                                      VectorXd& alpha) {
    auto& areaWeights = _area;
    auto& mArea = _area;

    // sanity check
    assert(p.size() > 0);
    assert((int)p.size() == alpha.size());

    assert(mArea.size() > 0);
    for (unsigned i = 0; i < p.size(); ++i)
    {
      //std::clog << p[i].size() << " ==? " << mArea.size() << std::endl;
      assert(p[i].size() == mArea.size());
      std::clog << (p[i].array() * mArea.array()).sum() << " ==? " << 1 << std::endl;
      std::clog << std::abs((p[i].array() * mArea.array()).sum() - 1.0) << std::endl;
      //assert(std::abs((p[i].array() * mArea.array()).sum() - 1.0) < 3.2e-5);
    }

    VectorXd q = p[0];

    // init
    const unsigned k = p.size();
    const unsigned n = mArea.size();
    const double opteps = 1e-300;

    alpha /= alpha.sum();
    // for (int i = 0; i < int(k); ++i) p[i].array() += opteps;

    q = VectorXd::Constant(n, 1.);
    std::vector<VectorXd> v(k, VectorXd::Constant(n, 1.));
    std::vector<VectorXd> w(k, VectorXd::Constant(n, 1.));

    int iter = 0;
    for ( ; iter < _niter; ++iter)
    {
      // update w
      // #pragma omp parallel for
      for (int i = 0; i < int(k); ++i)
      {
        VectorXd tmp = Kv(v[i]);
        clamp(tmp, opteps);
        w[i] = p[i].array() / tmp.array();
      }

      // compute auxiliar d
      std::vector<VectorXd> d(k);
      for (int i = 0; i < int(k); ++i)
      {
        d[i] = v[i].array() * Kv(w[i]).array();
        clamp(d[i], opteps);
      }

      // update barycenter q
      VectorXd prev = q;
      q = VectorXd::Zero(n);
      for (int i = 0; i < int(k); ++i)
      {
        q = q.array() + alpha[i] * d[i].array().log();
      }
      q = q.array().exp();

      // // Optional
      // if (useSharpening and iter > 0) sharpen(q);

      // update v
      for (int i = 0; i < int(k); ++i)
      {
        v[i] = v[i].array() * (q.array() / d[i].array());
      }

      // stop criterion
      const VectorXd diff = prev - q;
      const double res = dotProduct(diff, diff);
      // if (verbose > 0)
      //   std::cout << green << "Iter " << white << iter
      //             << ": " << res << std::endl;
      const double tol = 1e-7;
      if (iter > 1 and res < tol) break;
    }

    return q;
  };
};
