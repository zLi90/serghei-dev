#ifndef _PARTICLETRACKING_H_
#define _PARTICLETRACKING_H_

#include "define.h"
#include "tools.h"
#include "Indexing.h"
#include "Domain.h"
#include <cstdlib>
#include <random>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#if SERGHEI_LPT
#if SERGHEI_MESH_UNIFORM==1

//Tolerances
#define ZERO_VELOCITY 1E-6

//Initial particles distributions: (initialDist)
#define SERGHEI_PARTICLE_RANDOM_COORDINATES 100
#define SERGHEI_PARTICLE_RANDOM_CELLS 101
#define SERGHEI_PARTICLE_RANDOM_POLYGON 102
#define SERGHEI_PARTICLE_COORDINATES 103

//Initial Lifetime
#define SERGHEI_PARTICLE_RANDOM_LIFETIME 32
#define SERGHEI_PARTICLE_CONSTANT_LIFETIME 33

//Lifetime options
#define SERGHEI_PARTICLE_NO_RESURRECTION 11
#define SERGHEI_PARTICLE_RESURRECTION 12
#define SERGHEI_PARTICLE_RANDOM_RESURRECTION 13
#define SERGHEI_PARTICLE_RESURRECTION_PERIOD 14

class ParticleType{
  public:
    geometry::point x;
    //geometry::point v;
    real z;
    real t0;
    real lifetime;
    real life;
    real TotalLife;
    real TravelTime;
    real StopTime;
    real CoveredDistance;
    int checkBC;
    int active;
    int r;
    int index;
    int LifetimeOption1;
    int ii;
    int iGlob;
    real x1, y1;

    #if SERGHEI_LPT_MICROPLASTICS
      real Ra0;//Initial value of the radius particle
      real Ra;//Radius particle at the n time step. It is updated at each time step, because it will be reduced by the degradation.
      real L0;//Initial value of the length particle. It is only used for the cylinder form.
      real L;//Length particle at the n time step. It is updated at each time step, because it will be reduced by the degradation.
      real t0D;//initial time of degradation
    #endif
};


class ParserLine_Particles{
public:
  std::string line;
  std::string key ;
  std::stringstream value;

  void lowercase(){
    std::for_each(line.begin(), line.end(), [](char & c) {
      c = ::tolower(c);
    });
  }

  void parse(){
    // make sure key and value are clean (in case of reuse)
    key.clear();
    value.clear();
     // Remove spaces and tabs from the line
     // line.erase (std::remove(line.begin(), line.end(), ' '), line.end());
     // line.erase (std::remove(line.begin(), line.end(), '\t'), line.end());

     // If the line isn't empty and doesn't begin with a comment specifier, split it based on the colon
     if (!line.empty() && line.find("//",0) != 0) {

       // Find the colon
       uint splitloc = line.find(':',0);

       // Store the key and value strings
       key   = line.substr(0,splitloc);

       // Remove spaces and tabs from the key
       key.erase(std::remove(key.begin(), key.end(), ' '), key.end());
       key.erase(std::remove(key.begin(), key.end(), '\t'), key.end());
       std::string val = line.substr(splitloc+1,line.length()-splitloc);

       // Check for comments after values
       size_t splitter = val.find("//",0);
       std::string strloc;
       if (splitter != std::string::npos){
         strloc = val.substr(0,splitter);
       } else {
         strloc = val;
       }

       // Transform the val into a string stream for convenience
       value.clear();
       value.str(strloc);
     }
   }
};

typedef Kokkos::View<ParticleType*, Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::SharedSpace>> particleTypeView;

//Information of the cell where each cell is located
class particleSupport{
  public:
    geometry::point v;
    real dt;
    int edge;
    real dEdge;
    real xmin;
    real xmax;
    real ymin;
    real ymax;
};

//main class
class ParticleTracker{
  public:

    int N_par;  // number of particles
    int N_par_origin=1000000; //number of particles at the initial state
    real lifetimeIni;
    int initialDist; //type of initial distribution
    int initialLifetime; //initial lifetime
    int LifetimeOption;
    int numberPolygons;
    realArr pointsXPolygon;
    realArr pointsYPolygon;
    realArr pointsXPolygon2;
    realArr pointsYPolygon2;
    realArr Random_numbers;
    intArr Random_numbers_int;
    intArr index_BCcells;
    intArr index_Random;
    intArr CellsData;
    intArr IndexRandom;
    int Npoints;
    int Npoints2;
	  int Particle_nOut=1;
    int Par_period=1;

    particleTypeView particles;

    #if SERGHEI_LPT_DIFFUSIVE
      realArr R;
      int r=1;
      #if SERGHEI_LPT_MICROPLASTICS
        
        //Buoyancy variables
        real rho_p=930.0;
        real rho_w=1000.0;
        real nu=0.00000151;
        
        int body_form=1;
        /*PARTICLE FORM:
        1: sphere
        2: cylinder
        */

        //Degradation variables
        real tR=60.0;//degradation percent(%) per time unit (1% means that the particle loses 1% of its mass/volume per second).
      #endif
    #endif

    inline void initialisePolygon(){
      if(numberPolygons==1){//1 polygon of particles as intial condition
          pointsXPolygon = realArr("pointsXPolygon", Npoints);
          pointsYPolygon = realArr("pointsYPolygon", Npoints);
      }
      if(numberPolygons==2){//2 polygons of particles as intial condition
          pointsXPolygon = realArr("pointsXPolygon", Npoints);
          pointsYPolygon = realArr("pointsYPolygon", Npoints);
          pointsXPolygon2 = realArr("pointsXPolygon2", Npoints2);
          pointsYPolygon2 = realArr("pointsYPolygon2", Npoints2);
      }
    };

    inline void initialiseParticlesCoordinates(){
      particles = particleTypeView("particleTypeView", N_par);
    };

    inline void initialiseParticles(const Domain &dom, const State &state){

      #if SERGHEI_DEBUG_PARTICLES
        std::cout << GGD << GRAY << __PRETTY_FUNCTION__ << RESET << std::endl;
      #endif

      initalise(dom,state);
    };

    //////Functions to generate random numbers//////////////////////////////////

    KOKKOS_INLINE_FUNCTION double random_with_seed(unsigned int seed){
      unsigned int constant_a = 1664525;
      unsigned int constant_c = 1013904223;
      unsigned int max_value = 0xFFFFFFFF;
      unsigned int previous_value = seed;
      previous_value = (constant_a * previous_value + constant_c) % max_value;
      return static_cast<double>(previous_value) / static_cast<double>(max_value);
    };

    inline int getRandomNumber(int min, int max, int seed) {
        srand(seed);
        int numero_aleatorio=rand() % (max-min+1) + min;
        return numero_aleatorio;
    };

    KOKKOS_INLINE_FUNCTION double getRandomRealNumber(int seed){
      seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF;
      double randomNumber = static_cast<double>(seed) / 0x7FFFFFFF;
      return randomNumber;
    };
    ////////////////////////////////////////////////////////////////////////////

    ////Functions to initiliase each particle (position, lifetime,...)//////////
    inline void initalise(const Domain &dom, const State &state){
      Kokkos::Timer timer;
      // N needs to have been previously set
      int seed;
      int ip;
      seed=time(NULL);
      if(initialDist!=SERGHEI_PARTICLE_COORDINATES){
        particles = particleTypeView("particleTypeView", N_par);
      }
      srand(seed);
      Random_numbers = realArr( "Random_numbers", N_par);
      Random_numbers_int = intArr("Random_numbers_int", N_par);
      for(int i=0;i<N_par;i++){
        srand(seed);
        Random_numbers(i)=random_with_seed(seed+i);
      }
      #if SERGHEI_LPT_DIFFUSIVE
        R=realArr("R", N_par);
			  Kokkos::Random_XorShift64_Pool<> rand_pool(1234);
        Kokkos::parallel_for("RandomNumbers",N_par, KOKKOS_CLASS_LAMBDA(int i){
          auto rand_gen = rand_pool.get_state();
          R(i) = rand_gen.drand(-1.0, 1.0);
          rand_pool.free_state(rand_gen);
        });
        #if SERGHEI_LPT_MICROPLASTICS
          Kokkos::parallel_for("MicroplasticsInitialization",N_par, KOKKOS_CLASS_LAMBDA(int i){
            auto rand_gen2 = rand_pool.get_state();
            switch(body_form){
              case 1://SPHERE
                particles(i).Ra0=particles(i).Ra=0.001;
                break;
              case 2://CYLINDER
                particles(i).Ra0=particles(i).Ra=0.001;
                particles(i).L0=particles(i).L=0.001;
                break;
              default://UNKNOWN BODY FORM, IT IS CONSIDERED AS A SPHERE
                particles(i).Ra0=particles(i).Ra=0.001;
                break;
            }
            particles(i).t0D=rand_gen2.drand(0.0, 1.0)*dom.endTime;
            rand_pool.free_state(rand_gen2);
          });
        #endif
      #endif
      if(LifetimeOption!=SERGHEI_PARTICLE_RESURRECTION_PERIOD){
        N_par_origin=N_par;
      }
      if(N_par_origin>N_par){
          N_par_origin=N_par;
      }
      if(LifetimeOption==SERGHEI_PARTICLE_RESURRECTION_PERIOD || LifetimeOption==SERGHEI_PARTICLE_RESURRECTION){
        index_BCcells = intArr("index_BCcells", dom.nCellGlobal);
        ip=0;
        for(int ik=0;ik<dom.nCellGlobal;ik++){
          int iBC=dom.getIndex(ik);
          if(state.isBound(iBC)==1){
            index_BCcells(ip)=ik;
            ip++;
          }
        }
        index_Random = intArr("index_Random",N_par);
        for(int ikk=0;ikk<N_par;ikk++){
          srand(seed);
          index_Random(ikk)=getRandomNumber(0, ip,seed+ikk);
        }
      }
      for(int i=0;i<N_par_origin;i++){
        particles(i).t0 = 0.0;
        particles(i).TotalLife = 0.0;
        particles(i).TravelTime = 0.0;
        particles(i).StopTime = 0.0;
        particles(i).lifetime = 0.0;
        particles(i).CoveredDistance=0.0;
        particles(i).active=1;
        if(initialLifetime==SERGHEI_PARTICLE_CONSTANT_LIFETIME && particles(i).active==1){
          particles(i).life=lifetimeIni;
        }
        if(initialLifetime==SERGHEI_PARTICLE_RANDOM_LIFETIME && particles(i).active==1){
          particles(i).life=static_cast <float> (Random_numbers(i)) / (static_cast <float> (RAND_MAX/dom.simLength));
        }
      }
      if(LifetimeOption==SERGHEI_PARTICLE_RESURRECTION_PERIOD){
        for(int i=N_par_origin;i<N_par;i++){
          particles(i).t0 = 0.0;
          particles(i).lifetime = 0.;
          particles(i).active=0;
        } 
      }
      initialDistribution(dom,state);
      dom.timers.lpt.init = timer.seconds();

    };

    /*inline int point_in_polygon(real x, real y, realArr vertices_x, realArr vertices_y, int num_vertices){
      int i, j;
      int inside = 0;
      for (i = 0, j = num_vertices - 1; i < num_vertices; j = i++) {
        if (((vertices_y[i] > y) != (vertices_y[j] > y)) && (x < (vertices_x[j] - vertices_x[i]) * (y - vertices_y[i]) / (vertices_y[j] - vertices_y[i]) + vertices_x[i])){
          inside = !inside;
        }
      }
      return inside;
    };*/

    inline void initialDistribution(const Domain &dom, const State &state){ //, Geometry *geom){   // TODO, define a geometry class
      int seed,index,j,i;
      Kokkos::Random_XorShift64_Pool<> rand_pool2(1234);


      switch(initialDist){
        case SERGHEI_PARTICLE_RANDOM_CELLS: //the initial position for each cell is the center of a random cell.
          seed=time(NULL);
          CellsData = intArr("CellsData", dom.nCellGlobal-1);
          index=0;
          for(int ip=0;ip<dom.nCellGlobal;ip++){
            unpackIndicesUniformGrid(ip, dom.ny, dom.nx, j, i);
            int ii=(hc+j)*(dom.nx+2*hc)+hc+i;
            if(state.isnodata(ii)==0){
              CellsData(index)=ip;
              index++;
            }
          }
          IndexRandom=intArr("IndexRandom", N_par);
          for(int ijj=0;ijj<N_par;ijj++){
            srand(seed+ijj*20);
            IndexRandom(ijj)=rand() % index;
           }
          // TODO collect i and j for the 1D cells array
          Kokkos::parallel_for("initialiseCoordinates",N_par, KOKKOS_CLASS_LAMBDA(int ip){
            int i,j;
            int r = CellsData(IndexRandom(ip));
            unpackIndicesUniformGrid(r,dom.ny,dom.nx,j,i);
            particles(ip).x(_X) = (0.5+i)*dom.dx() + dom.extent[0](_X);
            particles(ip).x(_Y) = dom.extent[0](_Y) + dom.ny_glob*dom.dx() -(j+0.5)*dom.dx();
            int ii=(hc+j)*(dom.nx+2*hc)+hc+i;
            particles(ip).z = state.h(ii)+state.z(ii);
          });
          Kokkos::fence();
        break;

        case SERGHEI_PARTICLE_RANDOM_COORDINATES: //the initial position for each cell is a random coordinate
          seed=time(NULL);
          CellsData = intArr("CellsData", dom.nCellGlobal-1);
          index=0;
          for(int ip=0;ip<dom.nCellGlobal;ip++){
            unpackIndicesUniformGrid(ip, dom.ny, dom.nx, j, i);
            int ii=(hc+j)*(dom.nx+2*hc)+hc+i;
            if(state.isnodata(ii)==0){
              CellsData(index)=ip;
              index++;
            }
          }
          IndexRandom=intArr("IndexRandom", N_par);
          for(int ijj=0;ijj<N_par;ijj++){
            srand(seed+ijj*20);
            IndexRandom(ijj)=rand() % index;
          }
          // TODO collect i and j for the 1D cells array
          Kokkos::parallel_for("initialiseCoordinates",N_par, KOKKOS_CLASS_LAMBDA(int ip){
            int i,j;
            int r = CellsData(IndexRandom(ip));
            auto rand_gen = rand_pool2.get_state();
            real x = rand_gen.drand(0.0, dom.dx());
            rand_pool2.free_state(rand_gen);
            if(x<TOL12){
              x+=TOL8;
            }
            if(dom.dx()-x<TOL12){
              x-=TOL8;
            }
            real y = rand_gen.drand(0.0, dom.dx());
            rand_pool2.free_state(rand_gen);
            if(y<TOL12){
              y+=TOL8;
            }
            if(dom.dx()-y<TOL12){
              y-=TOL8;
            }
            unpackIndicesUniformGrid(r, dom.ny, dom.nx, j, i);
            particles(ip).x(_X) = i*dom.dx() + dom.extent[0](_X)+x;
            particles(ip).x(_Y) = dom.extent[0](_Y) + dom.ny_glob*dom.dx() -(j+1)*dom.dx()+y;
            int iGlob2 = dom.getCellForPoint(particles(ip).x);
            int ii2 = dom.getIndex(iGlob2);
            if(state.isnodata(ii2)!=0){
              particles(ip).x(_X) = (0.5+i)*dom.dx() + dom.extent[0](_X);
              particles(ip).x(_Y) = dom.extent[0](_Y) + dom.ny_glob*dom.dx() -(j+0.5)*dom.dx();
              iGlob2 = dom.getCellForPoint(particles(ip).x);
              ii2 = dom.getIndex(iGlob2);
            }
            #if SERGHEI_VERTICAL_VELOCITY || SERGHEI_LPT_MICROPLASTICS
              real z = rand_gen.drand(0.0, state.h(ii2));
              rand_pool2.free_state(rand_gen);
              if(z<TOL12){
                z+=TOL8;
              }
              if(state.h(ii2)-z<TOL12){
                z-=TOL8;
              }
              particles(ip).z = state.z(ii2) + z;
            #else
              particles(ip).z = state.h(ii2)+state.z(ii2);
            #endif
          });
          Kokkos::fence();
        break;

        case SERGHEI_PARTICLE_RANDOM_POLYGON:
          real xmin, xmax, ymin, ymax;
          real xmin2, xmax2, ymin2, ymax2;
          xmin=pointsXPolygon(0);
          xmax=pointsXPolygon(0);
          ymin=pointsYPolygon(0);
          ymax=pointsYPolygon(0);
          for(int pp=1;pp<Npoints;pp++){
            if(xmin>pointsXPolygon(pp)){
              xmin=pointsXPolygon(pp);
            }
            if(xmax<pointsXPolygon(pp)){
              xmax=pointsXPolygon(pp);
            }
            if(ymin>pointsYPolygon(pp)){
              ymin=pointsYPolygon(pp);
            }
            if(ymax<pointsYPolygon(pp)){
              ymax=pointsYPolygon(pp);
            }
          }
          if(numberPolygons==2){
              xmin2=pointsXPolygon2(0);
              xmax2=pointsXPolygon2(0);
              ymin2=pointsYPolygon2(0);
              ymax2=pointsYPolygon2(0);
              for(int pp2=1;pp2<Npoints2;pp2++){
                if(xmin2>pointsXPolygon2(pp2)){
                  xmin2=pointsXPolygon2(pp2);
                }
                if(xmax2<pointsXPolygon2(pp2)){
                  xmax2=pointsXPolygon2(pp2);
                }
                if(ymin2>pointsYPolygon2(pp2)){
                  ymin2=pointsYPolygon2(pp2);
                }
                if(ymax2<pointsYPolygon2(pp2)){
                  ymax2=pointsYPolygon2(pp2);
                }
              }
          }
          if(numberPolygons==1){
          for(int ip=0;ip<N_par;ip++){
              real x, y;
              do{
                  x=xmin+(rand()/(RAND_MAX/(xmax-xmin)));
                  y=ymin+(rand()/(RAND_MAX/(ymax-ymin)));
              //}while(!point_in_polygon(x, y, pointsXPolygon, pointsYPolygon, Npoints));
              }while(!geometry::isInsidePoly(Npoints,pointsXPolygon, pointsYPolygon, x, y));
              particles(ip).x(_X)=x;
              particles(ip).x(_Y)=y;
              int iGlob = dom.getCellForPoint(particles(ip).x);
              int ii = dom.getIndex(iGlob);
              particles(ip).z=state.h(ii)+state.z(ii);
              }
          }
          if(numberPolygons==2){
              int N_par_polygon;
              N_par_polygon=N_par/2;
              int rest;
              rest = N_par % 2;

              for(int ip=0;ip<(N_par_polygon+rest);ip++){
                  real x, y;
                  do{
                      x=xmin+(rand()/(RAND_MAX/(xmax-xmin)));
                      y=ymin+(rand()/(RAND_MAX/(ymax-ymin)));
                  //}while(!point_in_polygon(x, y, pointsXPolygon, pointsYPolygon, Npoints));
                  }while(!geometry::isInsidePoly(Npoints,pointsXPolygon, pointsYPolygon, x, y));
                  particles(ip).x(_X)=x;
                  particles(ip).x(_Y)=y;
                  int iGlob = dom.getCellForPoint(particles(ip).x);
                  int ii = dom.getIndex(iGlob);
                  particles(ip).z=state.h(ii)+state.z(ii);
              }

              for(int ip2=(N_par-N_par_polygon-rest);ip2<N_par;ip2++){
                  real x2, y2;
                  do{
                      x2=xmin2+(rand()/(RAND_MAX/(xmax2-xmin2)));
                      y2=ymin2+(rand()/(RAND_MAX/(ymax2-ymin2)));
                  //}while(!point_in_polygon(x2, y2, pointsXPolygon2, pointsYPolygon2, Npoints2));
                  }while(!geometry::isInsidePoly(Npoints2,pointsXPolygon2, pointsYPolygon2, x2, y2));
                  particles(ip2).x(_X)=x2;
                  particles(ip2).x(_Y)=y2;
                  int iGlob2 = dom.getCellForPoint(particles(ip2).x);
                  int ii2 = dom.getIndex(iGlob2);
                  particles(ip2).z=state.h(ii2)+state.z(ii2);
              }
          }

        break;

        case SERGHEI_PARTICLE_COORDINATES:

        break;

        default:
          seed=time(NULL);
          CellsData = intArr("CellsData", dom.nCellGlobal-1);
          index=0;
          for(int ip=0;ip<dom.nCellGlobal;ip++){
            unpackIndicesUniformGrid(ip, dom.ny, dom.nx, j, i);
            int ii=(hc+j)*(dom.nx+2*hc)+hc+i;
            if(state.isnodata(ii)==0){
              CellsData(index)=ip;
              index++;
            }
          }
          IndexRandom=intArr("IndexRandom", N_par);
          for(int i=0;i<N_par;i++){
            srand(seed+i*20);
            IndexRandom(i)=rand() % index;
          }
          // TODO collect i and j for the 1D cells array
          Kokkos::parallel_for("initialiseCoordinates",N_par, KOKKOS_CLASS_LAMBDA(int ip){
            int i,j;
            int r = CellsData(IndexRandom(ip));
            auto rand_gen = rand_pool2.get_state();
            real x = rand_gen.drand(0.0, dom.dx());
            rand_pool2.free_state(rand_gen);
            if(x<TOL12){
              x+=TOL8;
            }
            if(dom.dx()-x<TOL12){
              x-=TOL8;
            }
            real y = rand_gen.drand(0.0, dom.dx());
            rand_pool2.free_state(rand_gen);
            if(y<TOL12){
              y+=TOL8;
            }
            if(dom.dx()-y<TOL12){
              y-=TOL8;
            }
            unpackIndicesUniformGrid(r, dom.ny, dom.nx, j, i);
            particles(ip).x(_X) = i*dom.dx() + dom.extent[0](_X)+x;
            particles(ip).x(_Y) = dom.extent[0](_Y) + dom.ny_glob*dom.dx() -(j+1)*dom.dx()+y;
            int iGlob2 = dom.getCellForPoint(particles(ip).x);
            int ii2 = dom.getIndex(iGlob2);
            if(state.isnodata(ii2)!=0){
              particles(ip).x(_X) = (0.5+i)*dom.dx() + dom.extent[0](_X);
              particles(ip).x(_Y) = dom.extent[0](_Y) + dom.ny_glob*dom.dx() -(j+0.5)*dom.dx();
              iGlob2 = dom.getCellForPoint(particles(ip).x);
              ii2 = dom.getIndex(iGlob2);
            }
            particles(ip).z = state.h(ii2)+state.z(ii2);

          });
          Kokkos::fence();
        break;
      }
    };

    KOKKOS_INLINE_FUNCTION particleSupport getParticleSupport(const geometry::point &pos, const Domain &dom, const State &state) const{
      particleSupport ps;
      int iGlob = dom.getCellForPoint(pos);
      int ii = dom.getIndex(iGlob);
      real h = state.h(ii);
      real dx= dom.dx();
       // get velocity field
      ps.v(_X) = ps.v(_Y) = 0;

      if(h> state.hmin){
        ps.v(_X) = state.hu(ii)/h;
        ps.v(_Y) = state.hv(ii)/h;
      }

      // velocity magnitude
      real vmag = sqrt(ps.v(_X)*ps.v(_X) + ps.v(_Y)*ps.v(_Y));

      // if velocity is zero, early exit
      if(vmag < ZERO_VELOCITY){
        ps.dt=dom.dt;
        return(ps);
      }else{
        // unit vector of velocity
        geometry::point unitv = ps.v / vmag;

        // get southwest corner
        geometry::point SW;
        int kkk,lll;
        int Iglob = dom.getCellForPoint(pos);
        unpackIndicesUniformGrid(Iglob,dom.ny,dom.nx,kkk,lll);
        
        /*SW(_X) = i*dx;
        SW(_Y) = j*dx;*/
        
        SW(_X) = lll*dx+dom.extent[0](_X);
        ps.xmin=SW(_X);
        ps.xmax=ps.xmin+dx;
        //printf("X:%lf, i:%d, SW(_X):%lf, pos_x:%lf, v_x:%.12e\n",dom.extent[0](_X),lll,SW(_X),pos(_X),ps.v(_X));

        SW(_Y) = dom.extent[0](_Y) + dom.ny_glob*dx -(kkk+1)*dx;
        ps.ymin=SW(_Y);
        ps.ymax=ps.ymin+dx;
        //printf("Y:%lf, j:%d, SW(_Y):%lf, pos_y:%lf, v_y:%.12e\n",dom.extent[0](_Y),kkk,SW(_Y),pos(_Y),ps.v(_Y));

        const real LARGE_VALUE = dom.dx()*sqrt(2.0);

        real dEdge[N_CELL_EDGES];
        for(int kk=0; kk<N_CELL_EDGES; kk++) dEdge[kk] = LARGE_VALUE;
        if(unitv(_X) != ZERO_VELOCITY){
          // evaluate distance to intersection with east and west edges
          dEdge[_EAST] = (SW(_X) + dx - pos(_X))/unitv(_X);
          dEdge[_WEST] = (SW(_X) - pos(_X))/unitv(_X);
        }
        if(unitv(_Y) != ZERO_VELOCITY){
          // evaluate distance to intersection with north and south edges
          dEdge[_NORTH] = (SW(_Y) + dx - pos(_Y))/unitv(_Y);
          dEdge[_SOUTH] = (SW(_Y) - pos(_Y))/unitv(_Y);
        }

        // the shortest positive intersection distance is the actual intersection with cell edges
        // negative distances are antiparallel to the velocity vector, and should be discarded
        ps.dEdge = LARGE_VALUE;
        for(int kk=0; kk<N_CELL_EDGES; kk++){
          if(dEdge[kk]>0 && dEdge[kk] < ps.dEdge){
            ps.edge = kk;
            ps.dEdge = dEdge[kk];
          }
        }
        // time step to enforce particle to stay in cell
        ps.dt = ps.dEdge / vmag;
        if(ps.dt>dom.dt){
          ps.dt=dom.dt;
        }
        return(ps);
      }
     
    };

    //Function to obtain the temporal and spatial evolution of the cells////////
    inline void update(const Domain &dom, State &state){
      Kokkos::Timer timer;

      int newParticles=LifetimeOption;
      intArr Particles_per_period_2;

      int Particles_per_period;
      real Period;
      int resurrection_active;
      real tolerance;

      if(newParticles==SERGHEI_PARTICLE_RESURRECTION_PERIOD){
	      Particles_per_period=500;
	      Period=6000.0;
	      resurrection_active=0;
	      tolerance=0.01;

	      if((fabs(dom.etime - Period * (int)(dom.etime / Period)) < tolerance) || (fabs(dom.etime-Particle_nOut*Period)<TOL12)){
	          resurrection_active=1;
	      }else{
		      resurrection_active=0;
	      }
      }

      #if SERGHEI_LPT_DIFFUSIVE
        int seed;
        seed=time(NULL);
        srand(seed);
        int aleatorio, aleatorio2;
        aleatorio=rand() % (N_par+1);
        srand(seed);
        aleatorio2=rand() % (N_par+1);
      #endif

      timer.reset();

      real dx;
      real xll;
      real yll;
      real nx;
      real ny;
      real dt;
      dx=dom.dx();
      xll=dom.xll;
      yll=dom.yll;
      nx=dom.nx;
      ny=dom.ny;
      dt=dom.dt;

      Kokkos::parallel_for("updateParticles",N_par, KOKKOS_CLASS_LAMBDA(int i){
        int k,l;
        if(particles(i).life>0 && particles(i).active==1){
          real sumdtp=0.;
          real dtp=0.;

          ///////SINCHRONIZATION OF THE PARTICLES TIME STEP WITH THE 2D TIME STEP///////
          //The tolerance of the rest dt-sumpdtp must be equal than the tolerance imposed inside the while loop
          geometry::point x_old;
          x_old=particles(i).x;
          geometry::point pos =  x_old;
          geometry::point x_new = x_old;

          while(fabs(dt-sumdtp)>TOL8 && particles(i).life>0 && particles(i).active==1){
            
            if(sumdtp>0.0){
              pos=x_new;
            }

            particleSupport ps=getParticleSupport(pos,dom,state);
            dtp=min(ps.dt,dt-sumdtp);
            //The tolerance of the rest dt-sumpdtp must be equal than the tolerance imposed inside the while loop
            if(dtp<TOL8){
              dtp=dt-sumdtp;
              if(dtp<0.0){
                dtp=fabs(dtp);
              }
            }

            int iGlob0 = dom.getCellForPoint(x_new);
            int ii0 = dom.getIndex(iGlob0);
            for(int kk=0; kk<N_SPATIAL_DIM; kk++){
              x_new(kk)  = pos(kk) + dtp * ps.v(kk);
            }
            particles(i).CoveredDistance+=sqrt(dtp*ps.v(0)*dtp*ps.v(0)+dtp*ps.v(1)*dtp*ps.v(1));
            int iGlob01 = dom.getCellForPoint(x_new);
            int ii01 = dom.getIndex(iGlob01);

            int count=0;
            if((state.h(ii01)==0.0)  && (x_new(_X)>xll) && (x_new(_X)<(nx*dx+xll)) && (x_new(_Y)>yll) && (x_new(_Y)<(ny*dx+yll)) && state.isnodata(ii01)==0){
              x_new(_X)  -= dtp * ps.v(_X);
              x_new(_Y)  -= dtp * ps.v(_Y);
              particles(i).CoveredDistance-=sqrt(dtp*ps.v(0)*dtp*ps.v(0)+dtp*ps.v(1)*dtp*ps.v(1));
              while(state.h(ii01)==0.0 && count<5 && dtp>TOL8){
                dtp=dtp/2.0;
                x_new(_X)  += dtp * ps.v(_X);
                x_new(_Y)  += dtp * ps.v(_Y);
                particles(i).CoveredDistance+=sqrt(dtp*ps.v(0)*dtp*ps.v(0)+dtp*ps.v(1)*dtp*ps.v(1));
                iGlob01 = dom.getCellForPoint(x_new);
                ii01 = dom.getIndex(iGlob01);
                count++;
              }
            }
            

            ////////////////////TOLERANCES///////////////////
            if(x_new(_X)==ps.xmin && state.hu(ii01)<0.0){
              x_new(_X)-=TOL8;
            }
            if(x_new(_X)==ps.xmax && state.hu(ii01)>0.0){
              x_new(_X)+=TOL8;
            }
            if(x_new(_Y)==ps.ymin && state.hv(ii01)<0.0){
              x_new(_Y)-=TOL8;
            }
            if(x_new(_Y)==ps.ymax && state.hv(ii01)>0.0){
              x_new(_Y)+=TOL8;
            }
            ////////////////////////////////////////////////

            #if SERGHEI_VERTICAL_VELOCITY
              int iGlob_vertical = dom.getCellForPoint(x_new);
              int ii_vertical = dom.getIndex(iGlob_vertical);
              int count_vertical=0;
              if(particles(i).z<state.z(ii_vertical)){
                particles(i).CoveredDistance-=sqrt(dtp*ps.v(0)*dtp*ps.v(0)+dtp*ps.v(1)*dtp*ps.v(1));
                while(particles(i).z<state.z(ii_vertical) && count_vertical<5 && dtp>TOL8){
                  x_new(_X)  -= dtp * ps.v(_X);
                  x_new(_Y)  -= dtp * ps.v(_Y);
                  dtp=dtp/2.0;
                  x_new(_X)  += dtp * ps.v(_X);
                  x_new(_Y)  += dtp * ps.v(_Y);
                  particles(i).CoveredDistance+=sqrt(dtp*ps.v(0)*dtp*ps.v(0)+dtp*ps.v(1)*dtp*ps.v(1));
                  iGlob_vertical = dom.getCellForPoint(x_new);
                  ii_vertical = dom.getIndex(iGlob_vertical);
                  count_vertical++;
                }
              }
              iGlob_vertical = dom.getCellForPoint(x_new);
              ii_vertical = dom.getIndex(iGlob_vertical);
              if(particles(i).z<state.z(ii_vertical)){
                x_new(_X)  = particles(i).x(0);
                x_new(_Y)  = particles(i).x(1);
                particleSupport ps_vertical=getParticleSupport(x_new,dom,state);
                dtp=min(ps_vertical.dt,dt-sumdtp);
              } 
              int iGlob_oldz = dom.getCellForPoint(x_old);
              int ii_oldz = dom.getIndex(iGlob_oldz);
              #if SERGHEI_VERTICAL_VELOCITY_Z
                particles(i).z += 2.0*state.w(ii_oldz)*(particles(i).z-state.z(ii_oldz))/state.h(ii_oldz)*dtp;
              #else
                particles(i).z += state.w(ii_oldz)*dtp;
              #endif
              int iGlob_newz = dom.getCellForPoint(x_new);
              int ii_newz = dom.getIndex(iGlob_newz);
              if(particles(i).z>(state.h(ii_newz)+state.z(ii_newz))){
                particles(i).z=state.h(ii_newz)+state.z(ii_newz);
              }else if(particles(i).z<state.z(ii_newz)){
                particles(i).z = state.z(ii_newz);
                particles(i).life=0.0;
              }
            #endif

            ///Output Information/////////////
            if(ps.v(0) ==0.0 && ps.v(1)==0.0){
              particles(i).StopTime+=dtp;
            }else{
              particles(i).TravelTime+=dtp;
            }
            particles(i).TotalLife+=dtp;
            /////////////////////////////////

            #if SERGHEI_LPT_DIFFUSIVE
              int iGlob22 = dom.getCellForPoint(x_old);
              int ii22 = dom.getIndex(iGlob22);
              real h=state.h(ii22);
              real u = state.hu(ii22) / h;
              real v = state.hv(ii22) / h;
              real velocity_sq = u * u + v * v;
              real sqrt_velocity = sqrt(GRAV * velocity_sq / cbrt(h));
              real roughness = state.roughness(ii22);
              real epsilon_L = 0.01;
              real epsilon_T = 0.01;
              real epsilon_V = 0.067;
              
              if(h>state.hmin){

                //TURBULENCE IN X DIRECTION////////////////////////
                real K_hx=epsilon_L*h*roughness*sqrt_velocity;
                real sqrt_2_K_hx_dtp_r = sqrt(2 * K_hx * dtp / r);
                int index_aleatorio;

                if(aleatorio>i){
                  index_aleatorio=aleatorio-i;
                }else{
                  index_aleatorio=i-aleatorio;
                }
                x_new(_X) += R(index_aleatorio) * sqrt_2_K_hx_dtp_r;

                int iGlob2 = dom.getCellForPoint(x_new);//cell index of the new position, this is neccesary for the wet-dry problem
                int ii2 = dom.getIndex(iGlob2);
                if(x_new(_X)<xll || x_new(_X)>(nx*dx+xll) || state.h(ii2)<TOL9){
                  x_new(_X)  -= R(index_aleatorio)* sqrt_2_K_hx_dtp_r;
                }
                ///////////////////////////////////////////////
  

                //TURBULENCE IN Y DIRECTION////////////////////////
                real K_hy = epsilon_T * h * roughness * sqrt_velocity;
                real sqrt_2_K_hy_dtp_r = sqrt(2 * K_hy * dtp / r);
                
                if(i+aleatorio<N_par){
                  index_aleatorio=N_par-1-i-aleatorio;
                }else{
                  if(i>aleatorio){
                    index_aleatorio=N_par-1-i+aleatorio;
                  }else{
                    index_aleatorio=N_par-1+i-aleatorio;
                  }
                }
                x_new(_Y) += R(index_aleatorio) * sqrt_2_K_hy_dtp_r;

                int iGlob3 = dom.getCellForPoint(x_new);//cell index of the new position, this is neccesary for the wet-dry problem
                int ii3 = dom.getIndex(iGlob3);
                if(x_new(_Y)<yll || x_new(_Y)>(ny*dx+yll) || state.h(ii3)<TOL9){
                  x_new(_Y)  -= R(index_aleatorio)*sqrt(2*K_hy*dtp/r);
                }
                ///////////////////////////////////////////////
              }
            #endif

            if( (xll>=x_new(_X)) || (x_new(_X)>=(nx*dx+xll)) || (x_new(_Y)<=yll) || (x_new(_Y)>=(ny*dx+yll)) ){
                particles(i).active=0;
                particles(i).life=0.0;
            }else{
              int iGlob = dom.getCellForPoint(x_new);
              int ii = dom.getIndex(iGlob);
              if(state.isnodata(ii)!=0){
                particles(i).active=0;
                particles(i).life=0.0;
              }else{
                #if SERGHEI_LPT_DIFFUSIVE
                  if(particles(i).z>state.h(ii)+state.z(ii)){
                    particles(i).z=state.h(ii)+state.z(ii);
                  }
                  //TURBULENCE IN Z DIRECTION////////////////////////
                  real K_v=epsilon_V*h*roughness*sqrt_velocity;
                  real sqrt_2_K_v_dtp_r = sqrt(2 * K_v * dtp / r);
                  int index_aleatorio2;
                  if(u>0.009 || v>0.009){//if the velocity is greater than 0.009 m/s, there is turbulence in the vertical direction 
                    if(aleatorio2>i){
                      index_aleatorio2=aleatorio2-i;
                    }else{
                      index_aleatorio2=i-aleatorio2;
                    }
                    particles(i).z += R(index_aleatorio2)*sqrt_2_K_v_dtp_r;
                  }
                  /////////////////////////////////////////////// 
                  #if SERGHEI_LPT_MICROPLASTICS
                    real d_star;
                    /////////////////SETTLING VELOCITY///////////////////////
                    switch(body_form){
                      case 1://SPHERE
                        d_star=2.0*particles(i).Ra*cbrt(GRAV*(Kokkos::fabs(rho_p-rho_w))/(rho_w*nu*nu));
                        if(rho_p-rho_w<0){
                          particles(i).z+=nu/(2.0*particles(i).Ra)*d_star*d_star*d_star*pow((38.1+0.93*pow(d_star,12.0/7.0)),-7.0/8.0)*dtp;
                        }else{
                          particles(i).z-=nu/(2.0*particles(i).Ra)*d_star*d_star*d_star*pow((38.1+0.93*pow(d_star,12.0/7.0)),-7.0/8.0)*dtp;
                        }
                        break;
                      case 2://CYLINDER
                        break;
                      default://UNKNOWN BODY FORM, IT IS CONSIDERED AS A SPHERE
                        d_star=2.0*particles(i).Ra*cbrt(GRAV*(rho_p-rho_w)/rho_w*nu*nu);
                        particles(i).z-=nu/(2.0*particles(i).Ra)*d_star*d_star*d_star*pow((38.1+0.93*pow(d_star,12.0/7.0)),-7.0/8.0)*dtp;
                        break;
                    }
                    ///////////////////////

                    //////DEGRADATION//////
                     switch(body_form){
                      case 1://SPHERE
                        if(dom.etime>particles(i).t0D){//degradation starts
                          particles(i).Ra=particles(i).Ra0*(1-tR*(dom.etime-particles(i).t0D)/100);
                        }
                        if(particles(i).Ra<0.0000001){//degradation ends, particle disappears
                          particles(i).life=0.0;
                          particles(i).active=0;
                        }
                        break;
                      case 2://CYLINDER
                        break;
                      default://UNKNOWN BODY FORM, IT IS CONSIDERED AS A SPHERE
                        if(dom.etime>particles(i).t0D){//degradation starts
                          particles(i).Ra=particles(i).Ra0*(1-tR*(dom.etime-particles(i).t0D)/100);
                        }
                        if(particles(i).Ra<0.0000001){//degradation ends, particle disappears
                          particles(i).life=0.0;
                          particles(i).active=0;
                        }
                        break;
                    }
                    ///////////////////////

                    if(particles(i).z>state.h(ii)+state.z(ii)){
                      particles(i).z = state.h(ii)+state.z(ii);
                    }
                    if(particles(i).z<state.z(ii) || particles(i).z==state.z(ii)){//deposition
                      particles(i).z=state.z(ii);
                      particles(i).life=0.0;
                      particles(i).active=0;
                    }
                  #else
                    if(particles(i).z>state.h(ii)+state.z(ii)){
                      particles(i).z -= R(index_aleatorio2)*pow(2*K_v*dtp/r,1.0/2.0);
                    }
                    if(particles(i).z<state.z(ii)){//deposition
                      particles(i).z=state.z(ii);
                      particles(i).life=0.0;
                      particles(i).active=0;
                    }
                  #endif
                #else
                  #if SERGHEI_VERTICAL_VELOCITY == 0
                    particles(i).z=state.h(ii)+state.z(ii);
                  #endif
                #endif
                int iGlobFinal = dom.getCellForPoint(x_new);
                int iiFinal = dom.getIndex(iGlobFinal);
                if(state.isBound(iiFinal)<0){
                  particles(i).checkBC=1;
                }
                if(particles(i).checkBC==1 && (fabs(x_new(_X)-xll)<TOL6 || fabs(xll-x_new(_X))<TOL6 || fabs(x_new(_Y)-yll)<TOL6 || fabs(yll-x_new(_Y))<TOL6)){
                  particles(i).life=0.0;
                  particles(i).active=0;
                }else{
                  sumdtp+=dtp;
                }
              }
            }
          }
          particles(i).lifetime += dt;
          particles(i).life-=dt;
          if(particles(i).active==1){
            particles(i).x(_X)  = x_new(_X);
            particles(i).x(_Y)  = x_new(_Y);
          }
        }else{
          //The particle's life is null, so the particle doesn't have movement
          #if SERGHEI_VERTICAL_VELOCITY==0
            particles(i).active=0;
          #endif
          int iindex=i;
          if(newParticles==SERGHEI_PARTICLE_RESURRECTION){
			      int r = index_Random(iindex);
            int index=index_BCcells(r);
            int ii2 = dom.getIndex(index);
            unpackIndicesUniformGrid(index, dom.ny, dom.nx, k, l);
            particles(i).active=1;

            if(initialLifetime == SERGHEI_PARTICLE_RANDOM_LIFETIME){
              particles(i).life=static_cast <float> (Random_numbers(i)) / (static_cast <float> (RAND_MAX/dom.simLength));
            }else{
              particles(i).life=lifetimeIni;
            }
			      particles(i).x(_X) = l*dx + dom.extent[0](_X)+i*dx/N_par;
			      particles(i).x(_Y) = dom.extent[0](_Y) + dom.ny_glob*dx -(k+1)*dx+i*dx/N_par;
            particles(i).z=state.h(ii2)+state.z(ii2);

          }
          if(newParticles==SERGHEI_PARTICLE_RANDOM_RESURRECTION){
            int i1,j1;
            int r1 = Random_numbers_int(i);
            real x11 = Random_numbers(i);
            if(x11<TOL12){
              x11+=TOL9;
            }
            real y11= Random_numbers(N_par-1-i);
            if(y11<TOL12){
              y11+=TOL9;
            }
            int ii=dom.getIndex(r1);
            unpackIndicesUniformGrid(r1, dom.ny, dom.nx, j1, i1);
            particles(i).x(_X) = xll + i1*dx + x11*dx;
            particles(i).x(_Y) = yll + j1*dx + y11*dx;
            particles(i).z = state.h(ii)+state.z(ii);
            particles(i).active=1;
            if(initialLifetime == SERGHEI_PARTICLE_RANDOM_LIFETIME){
              particles(i).life=static_cast <float> (Random_numbers(i)) / (static_cast <float> (RAND_MAX/dom.simLength));
            }else{
              particles(i).life=lifetimeIni;
            }
          }
          if(newParticles==SERGHEI_PARTICLE_RESURRECTION_PERIOD && i/Par_period<Particles_per_period && resurrection_active==1){
            int r = index_Random(iindex);
            int index=index_BCcells(r);
            int ii2 = dom.getIndex(index);
            unpackIndicesUniformGrid(index, dom.ny, dom.nx, k, l);
            particles(i).active=1;
            if(initialLifetime == SERGHEI_PARTICLE_RANDOM_LIFETIME){
              particles(i).life=static_cast <float> (Random_numbers(i)) / (static_cast <float> (RAND_MAX/dom.simLength));
            }else{
              particles(i).life=lifetimeIni;
            }
			      particles(i).x(_X) = l*dx + dom.extent[0](_X)+i*dx/N_par;
			      particles(i).x(_Y) = dom.extent[0](_Y) + dom.ny_glob*dx -(k+1)*dx+i*dx/N_par;
            particles(i).z=state.h(ii2)+state.z(ii2);

          }
        }
      });//Kokkos_for
      Kokkos::fence();

      ///////Output Information:t_particles////////////////////
      Kokkos::parallel_for("t_particle_compute", N_par , KOKKOS_CLASS_LAMBDA(int i_par) {
        if(particles(i_par).active==1){
          int iGlob_par = dom.getCellForPoint(particles(i_par).x);
          int ii_par = dom.getIndex(iGlob_par);
          if(state.market(ii_par)==0){
            state.market(ii_par)=1;
            state.tparticles(ii_par)+=dom.dt;
            if(state.tparticles(ii_par)>dom.etime){
              state.tparticles(ii_par)=dom.etime;
            }
          }
        }
      });
      Kokkos::fence();

      Kokkos::parallel_for("state_market_initialization", dom.nCell , KOKKOS_LAMBDA(int iGlob_state) {
        state.market(iGlob_state)=0;
      });
      Kokkos::fence();
      ////////////////////////////////////////////////////////

      if(newParticles==SERGHEI_PARTICLE_RESURRECTION_PERIOD){
	      if((fabs(dom.etime - Period * (int)(dom.etime / Period)) < tolerance) || (fabs(dom.etime-Particle_nOut*Period)<TOL12)){
			    Particle_nOut++;
		    }
        if(resurrection_active==1){
          Par_period++;
        }
      }
      dom.timers.lpt.update += timer.seconds();
    };

    /*KOKKOS_INLINE_FUNCTION void ProcessParticle(const geometry::point &pos, const Domain &dom, const State &state, const Parallel &par) const{
      Kokkos::parallel_for("updateParticles",N_par, KOKKOS_CLASS_LAMBDA(int i){
        if(particles(i).x(_X)<dom.extend[0](_X) || particles(i).x(_X)>dom.extend[1](_X) || particles(i).x(_Y)<dom.extend[0](_Y) || particles(i).x(_Y)>dom.extend[1](_Y)){

        }else{
                  particles(i).process=par.rank;
              }
          }
      };*/

};
#endif
#endif
#endif
