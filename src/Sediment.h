#ifndef _SEDIMENT_H_
#define _SEDIMENT_H_

#if SERGHEI_SEDIMENT_TRANSPORT
#include "define.h"
#include "ScalarTransport.h"
#include <numeric>

#define SERGHEI_RAIN_DETACHMENT 1
//erosion models
#define SERGHEI_FLOW_EROSION 1
#define SERGHEI_RAINFALL_EROSION 2

#define MOMENTUM_DISSIPATION 1
#define momReductionFactor 0.8

#define TOLSB1 1e-4
#define TOLSB2 1e-2

class SedimentTransport; 
class SedimentSolver; 

class SedimentTransport{

public:
  std::string dir;
  std::string fname="sediment.input";
  int nSed=0;
  std::set<std::string> initialModes = {"none","constant","file","netcdf"};
  std::vector<std::string> sedNames;  
  std::vector< SArray<real,2> > diffc;
  std::vector<real> initialValue;
  std::vector<real> dsValue; 
  std::vector<real> bedFractionValue; 
  std::vector<real> entrainmentCoefValue;
  std::vector<real> depositionCoefValue;        
  int sedMode = 0; 
  real bedStabilityAngle=-1.;
  real erodibleLayerThickness=-1.;
  int erosionModel = -1;
  real interrillDetachability=-1.;
  real rillErodibilityCoeff=-1.;
  real wsTurbulenceCoeff=-1.;

  std::string initialMode="none";  

  int read(const Parallel &par){
    #if SERGHEI_DEBUG_SEDIMENT
      std::cout << GGD << __PRETTY_FUNCTION__ << std::endl;
    #endif
    std::string fNameIn = dir + fname;
    std::ifstream fInStream(fNameIn);
    std::string line;
    ParserLine pline;
    std::string tmpStr;
    int sedcount=0;
    real accum;

    if (fInStream.is_open()){
      while (std::getline(fInStream, line)) {
        pline.line = line;
        pline.lowercase();
        pline.parse();

        if(!pline.key.empty()){
          if(!strcmp("nsediment",pline.key.c_str())){
            pline.value >> nSed;
            std::cout << BDASH "Number of sediments to transport: " << nSed << std::endl;
          }
          if(!strcmp("initialmode",pline.key.c_str())){
            pline.value >> initialMode;
          }
          if(!strcmp("bedstabilityangle",pline.key.c_str())){
            pline.value >> bedStabilityAngle;
          }
          if(!strcmp("erodiblelayerthickness",pline.key.c_str())){
            pline.value >> erodibleLayerThickness;
          }  
          if(!strcmp("erosionmodel",pline.key.c_str())){
            pline.value >> erosionModel;
            std::cout << BDASH "Erosion model: " << erosionModel << std::endl;
          }
          if(!strcmp("interrilldetachability",pline.key.c_str())){
            pline.value >> interrillDetachability;
            std::cout << BDASH "Interrill detachability " << interrillDetachability << std::endl;
          }  
          if(!strcmp("rillerodibilitycoeff",pline.key.c_str())){
            pline.value >> rillErodibilityCoeff;
            std::cout << BDASH "Rill erodibility coeff: " << rillErodibilityCoeff << std::endl;
          }  
          if(!strcmp("settlingturbulencecoeff",pline.key.c_str())){
            pline.value >> wsTurbulenceCoeff;
            std::cout << BDASH "ws turbulence coeff: " << wsTurbulenceCoeff << std::endl;
          }  

          //----- 
          if(nSed>0){
            if(!strcmp("sedimentname",pline.key.c_str())){ 
              pline.value >> tmpStr;
              sedNames.push_back(tmpStr);
              sedcount++;
            }
            if(!strcmp("initialvalue",pline.key.c_str())){ 
              real temp;
              pline.value >> temp; 
              initialValue.push_back(temp); 
            }
            if(!strcmp("dsvalue",pline.key.c_str())){ 
              real temp;
              pline.value >> temp; 
              dsValue.push_back(temp); 
            }
            if(!strcmp("bedfraction",pline.key.c_str())){ 
              real temp;
              pline.value >> temp; 
              bedFractionValue.push_back(temp); 
            }
            if(!strcmp("entrainmentcoef",pline.key.c_str())){ 
              real temp;
              pline.value >> temp; 
              entrainmentCoefValue.push_back(temp); 
            } 
            if(!strcmp("depositioncoef",pline.key.c_str())){ 
              real temp;
              pline.value >> temp; 
              depositionCoefValue.push_back(temp); 
            }                                                     
            if(!strcmp("diffc",pline.key.c_str())){
              SArray<real,2> tmp;
              pline.value >> tmp(0) >> tmp(1);
              diffc.push_back(tmp);
            }
          }
        }
      }
      if (sedcount<nSed){
        if(par.masterproc) std::cerr << RERROR "Sediment number mismatch. Expected " << nSed << " sediment defintions but only " << sedcount << " found. " << std::endl;
        return 0;
      }
      accum = std::reduce(bedFractionValue.begin(), bedFractionValue.end());
      if(accum != 1.){
        if(par.masterproc) std::cerr << RERROR "Non-physical bed fraction definitions. The sum of the individual bed fractions should be 1, but is " << accum << ". Check all sediment block definitions" << std::endl; 
        return 0;
      }
    }else{
      if (par.masterproc){
        std::cerr << RERROR "File " << fNameIn << " not found" << std::endl;
        return 0;
      }
    }
    if(nSed > 0) sedMode = 1;

    if(sedNames.size() < nSed || diffc.size() < nSed){
      std::cout << RERROR "Inconsistent declared number of sediments with sediment provided in " << fNameIn << std::endl;
      std::cout << "Number of sediments declared: " << nSed << std::endl;
      std::cout << "Defined sediment names: " << sedNames.size() << std::endl;
      std::cout << "Defined sediment diameters: " << dsValue.size() << std::endl;    
      std::cout << "Defined sediment fractions: = " << bedFractionValue.size() << std::endl;    
      std::cout << "Defined sediment diffusion coefficients: " << diffc.size() << std::endl;
      return(0);
    }    

    std::cout << GOK "Sediment transport input read" << std::endl;
    return 1;
  };

  int addSedToScalarTransport(ScalarTransport &st){
			st.nScalar += nSed;
			for(int i=0;i<nSed;i++){
			  st.scalarNames.push_back(sedNames[i]);
        st.initialValue.push_back(initialValue[i]);
      }
			if(st.initialMode.compare("none") && initialMode.compare(st.initialMode)){
				std::cerr << RERROR "Sediment initial mode does not match with scalar initial mode" << std::endl;
				return 0;
			}else{
        st.initialMode = initialMode;
      }
      return 1;   
  }

};


class SedimentSolver{

public:

  int iphised;
  int nSed;
  int erosionModel=0; //default model is 0: no-erosion
  int movableBedLevel=1; //movable bed activated by default

  realArr ds; //sediment diameter
  realArr fbed; //sediment bed fraction
  realArr ws; //sediment settling velocity
  realArr sedCharVel; //sediment characteristic velocity

  realArr entrainCoef; //entraenment coefficient
  realArr depositCoef; //deposition coefficient

  real dsmax, dsmean;
  real poros, bedConc, rhob; //bed porosity, bed solid concentration, bed saturated density
  real shieldsC; //critical Shields stress
  real stabAngle=75.0; //Maximun collapse angle
  real tanDeltaf; //tan(stabAngle)

  realArr bedExchangeVol;
  realArr FVCover; //Fraction of Vegetation Cover (FVC)
  realArr LUcoeff; //Land management practices coefficient
  realArr ZBthick; //thickness of the erodible soil layer

  realArr dz0;
  realArr dz1; 

  real kinterrill=0.0; //interrill erodibility
  real Krill=0.0; //rill detachment efficiency
  real BTrill=0.0; //turbulent mixing coefficient
  real zbt0=10.0; //default thickness of the erodible soil layer (practical no limitation)

	void addSeds(const int n, ADEsolver &ade){
		ade.nScalar += n;
    ade.nSed = n;
	}  

	int readFractionVegetationCover(std::string fNameIn, Domain &dom, const Parallel &par, const SedimentTransport &sedt){  
      #if SERGHEI_DEBUG_SEDIMENT
      std::cout << GGD << "Reading FVCover field from " << fname << std::endl;
      #endif

			realArr buffer = realArr("subview_buffer",dom.nCellMem);

      if(!readRasterField(fNameIn,dom,par,buffer)){
        if (par.masterproc){
          std::cerr << YEXC << fNameIn << " not found" << std::endl; 
          std::cerr << BDASH "Fractional vegetation cover set to uniform null" << std::endl;
        }	        
        return 0;
      }

      Kokkos::parallel_for("subview_buffer", dom.nCell , KOKKOS_CLASS_LAMBDA (int iGlob) {
        int ii = dom.getIndex(iGlob);
        FVCover(ii) = buffer(ii);
      });

		return 1;
	}   


	int readLandManagementFactor(std::string fNameIn, Domain &dom, const Parallel &par, const SedimentTransport &sedt){  
      #if SERGHEI_DEBUG_SEDIMENT
      std::cout << GGD << "Reading LUcoeff field from " << fname << std::endl;
      #endif

			realArr buffer = realArr("subview_buffer",dom.nCellMem);

      if(!readRasterField(fNameIn,dom,par,buffer)){
        if (par.masterproc){
          std::cerr << YEXC << fNameIn << " not found" << std::endl; 
          std::cerr << BDASH "Management factor set to uniform 1.0" << std::endl;
        }	        
        return 0;
      }

      Kokkos::parallel_for("subview_buffer", dom.nCell , KOKKOS_CLASS_LAMBDA (int iGlob) {
        int ii = dom.getIndex(iGlob);
        LUcoeff(ii) = buffer(ii);
      });

		return 1;
	}  


	int readErodibleBedThickness(std::string fNameIn, Domain &dom, const Parallel &par, const SedimentTransport &sedt){  
      #if SERGHEI_DEBUG_SEDIMENT
      std::cout << GGD << "Reading ZBthink field from " << fname << std::endl;
      #endif

			realArr buffer = realArr("subview_buffer",dom.nCellMem);

      if(!readRasterField(fNameIn,dom,par,buffer)){
        if (par.masterproc){
          std::cerr << YEXC << fNameIn << " not found" << std::endl; 
          std::cerr << BDASH "Erodible thickness ZBthick set to uniform user-defined ZBT0" << std::endl;
        }	        
        return 0;
      }

      Kokkos::parallel_for("subview_buffer", dom.nCell , KOKKOS_CLASS_LAMBDA (int iGlob) {
        int ii = dom.getIndex(iGlob);
        if(buffer(ii)>=0.0 && buffer(ii)<=10.0){
          ZBthick(ii) = buffer(ii);
        }
      });

		return 1;
	}     


  void initialise(Domain &dom, const Parallel &par, const ScalarTransport &st, const SedimentTransport &sedt){
    #if SERGHEI_DEBUG_SEDIMENT
      std::cout << GGD << __PRETTY_FUNCTION__ << std::endl;
    #endif 
    //-----------------------------------------------------  
    std::string fname;
    nSed = sedt.nSed;
    iphised = st.nScalar-sedt.nSed; //first sediment index, here nScalar already has nSed included

    if(sedt.bedStabilityAngle>TOL12 && sedt.bedStabilityAngle<75.){
      stabAngle=sedt.bedStabilityAngle;
    }
    if(sedt.erosionModel>=1 && sedt.erosionModel<=2){
      erosionModel=sedt.erosionModel;
    }
    if(sedt.erodibleLayerThickness>=0. && sedt.erodibleLayerThickness<=10.){
      zbt0=sedt.erodibleLayerThickness;
    }    
    if(sedt.interrillDetachability>=0. && sedt.interrillDetachability<=10.){
      kinterrill=sedt.interrillDetachability;
    }
    if(sedt.rillErodibilityCoeff>=0. && sedt.rillErodibilityCoeff<=100.){
      Krill=sedt.rillErodibilityCoeff;
    }
    if(sedt.wsTurbulenceCoeff>=0. && sedt.wsTurbulenceCoeff<=100.){
      BTrill=sedt.wsTurbulenceCoeff;
    }                            

    //allocate sediment views
    ds = realArr("ds", sedt.nSed);
    fbed = realArr("fbed", sedt.nSed);
    ws = realArr("ws", sedt.nSed);
    sedCharVel = realArr("sedCharVel", sedt.nSed);
    entrainCoef = realArr("entrainCoef", sedt.nSed);
    depositCoef = realArr("depositCoef", sedt.nSed);

    real aux1, aux2;
    dsmax=0.0;
    dsmean=0.0;   
    for(int ised=0; ised<sedt.nSed; ised++){
      ds(ised) = sedt.dsValue[ised];
      fbed(ised) = sedt.bedFractionValue[ised];

      dsmax=fmax(dsmax,ds(ised));
      dsmean+=fbed(ised)*ds(ised);

      sedCharVel(ised) = sqrt((RHOS-RHOW)/RHOW*GRAV*ds(ised));
      //ws(ised) = 1.72*sedCharVel(ised);
      aux1=13.95*VISCNU/ds(ised);
      aux2=1.09*sedCharVel(ised)*sedCharVel(ised);
      ws(ised) = sqrt(aux1*aux1 + aux2) - aux1; 

      entrainCoef(ised) = sedt.entrainmentCoefValue[ised];
      depositCoef(ised) = sedt.depositionCoefValue[ised];
    }

    poros=0.13 + 0.21/mypow(0.002+1000.*dsmean , 0.21); //bulk porosity by WU formaula
    bedConc=1.-poros;
    rhob=RHOW+(RHOS-RHOW)*bedConc;
    shieldsC=0.050;  
    tanDeltaf=tan(stabAngle * PI/180.0);

    //------------------------------------------------------------------------------
    bedExchangeVol = realArr("bedExchangeVol", dom.nCellMem);
    FVCover = realArr("FVCover", dom.nCellMem);
    LUcoeff = realArr("LUcoeff", dom.nCellMem);
    ZBthick = realArr("ZBthick", dom.nCellMem);

    dz0 = realArr("dz0", dom.nCellMem);
    dz1 = realArr("dz1", dom.nCellMem);
    //minSbk = realArr("minSbk", dom.nCellMem);    

    Kokkos::parallel_for("FVCover",dom.nCell, KOKKOS_CLASS_LAMBDA(int iGlob){
      int ii = dom.getIndex(iGlob);
      bedExchangeVol(ii) = 0.0;
      FVCover(ii) = 0.0; //default max erosion
      LUcoeff(ii) = 1.0; //default max erosion
      ZBthick(ii) = zbt0; //default uniform limit

      dz0(ii) = 0.0;
      dz1(ii) = 0.0;

    });

    fname = sedt.dir + "FVCover.input";
    if(readFractionVegetationCover(fname,dom,par,sedt)){
      std::cout << GOK << "FVCover field read from file " << fname << std::endl; 
    } 

    fname = sedt.dir + "LUcoeff.input";
    if(readLandManagementFactor(fname,dom,par,sedt)){
      std::cout << GOK << "LUcoeff field read from file " << fname << std::endl; 
    }  

    fname = sedt.dir + "ZBthick.input";
    if(readErodibleBedThickness(fname,dom,par,sedt)){
      std::cout << GOK << "ZBthick field read from file " << fname << std::endl; 
    }          

  };

  #if SERGHEI_SUSPENDED_SEDIMENT
  KOKKOS_INLINE_FUNCTION void computeSuspendedSedimentExchange(const int ii, 
    const real &h, 
    const real &hu, 
    const real &hv, 
    const real &roughness, 
    ADEsolver &ade,
    const real &hmin,
    const real &dt,
    const real &rainRate,
    const real &dzb){
    #if SERGHEI_DEBUG_SEDIMENT>1
      std::cout << GGD << __PRETTY_FUNCTION__ << std::endl;
    #endif
    //----------------------------------------------------- 
    real z0=0.0;
    real qflow=0.0; 
    real umod=0.0;
    real Sf=0.0; 
    real tau=0.0;  
    
    //reset sediment source term neccssary in all cells
    for(int iphi=iphised; iphi<(iphised+nSed); iphi++){
      ade.hphiSource(ii,iphi) = 0.0; 
    }    

    if(h>TOL12){ //wet cells
      z0=fmax(dsmax,hmin);
      if(h > z0){
        qflow=sqrt(hu*hu+hv*hv);
        umod=qflow/h;
        Sf=roughness*roughness*umod*umod/(h*cbrt(h));
        tau=RHOW*GRAV*h*Sf;
      } 

	  	switch (erosionModel){
        case SERGHEI_FLOW_EROSION: 
          fractionalErosionRiver(ii,h,qflow,umod,tau,z0,dt,ade,dzb); 
          break;   
        case SERGHEI_RAINFALL_EROSION:  
          fractionalErosionRainfall(ii,h,qflow,umod,tau,z0,dt,rainRate,ade,dzb);          
          break; 
      	default:
          break;           
      }
    }

  };
  #endif  


  #if SERGHEI_SUSPENDED_SEDIMENT
  KOKKOS_INLINE_FUNCTION void fractionalErosionRiver(const int ii, 
    const real &h, 
    const real &qflow,
    const real &umod,  
    const real &tau, 
    const real &z0,
    const real &dt,  
    ADEsolver &ade,
    const real &dzb){
    #if SERGHEI_DEBUG_SEDIMENT>1
      std::cout << GGD << __PRETTY_FUNCTION__ << std::endl;
    #endif
    //----------------------------------------------------- 
    real dsi, fbedi, wsi, sedCharVeli, alphaEi, alphaDi;
    real tauC;
    real phi;
    real qscap;
    real aux;
    real sedVol = 0.0;
    
    //available erodible thickness
    real tz;
    tz = ZBthick(ii) + dzb; 
    if(tz<0.0) tz=0.0;

    for(int iphi=iphised; iphi<(iphised+nSed); iphi++){
      dsi=ds(iphi-iphised);
      fbedi=fbed(iphi-iphised);
      wsi=ws(iphi-iphised);
      sedCharVeli=sedCharVel(iphi-iphised);
      alphaEi=entrainCoef(iphi-iphised);
      alphaDi=depositCoef(iphi-iphised);

      //deposition rate
      phi=ade.hphi(ii,iphi)/h; 
      sedVol = alphaDi * wsi * phi * dt; //use current dt for limiting, which can be different to real dt
      if(sedVol > ade.hphi(ii,iphi)){ //limit deposition
        sedVol = ade.hphi(ii,iphi);
      }
      ade.hphiSource(ii,iphi) -= sedVol/dt;         

      //entrainment rate
      if(h > z0){
        sedVol=0.0;

        if(tau > TOL12){
          tauC=shieldsC*(RHOS-RHOW)*GRAV*dsi;
          aux=2.62e-5*mypow((tau/tauC * umod/wsi),1.74);  
          qscap=aux*fbedi*dsi*sedCharVeli;

          //limiting qscap by flow discharge
          aux=fbedi*bedConc*qflow;
          if(qscap > aux) qscap = aux;

          //entrainment rate
          sedVol += alphaEi * wsi * qscap/qflow * dt;

          // erodible layer thikness limitation
          if(sedVol > fbedi*bedConc*tz){
            sedVol = fbedi*bedConc*tz;
          }

          ade.hphiSource(ii,iphi) += sedVol/dt; 
        }
      } 

    } 

  };
  #endif


  #if SERGHEI_SUSPENDED_SEDIMENT
  KOKKOS_INLINE_FUNCTION void fractionalErosionRainfall(const int ii, 
    const real &h, 
    const real &qflow,
    const real &umod,  
    const real &tau, 
    const real &z0,
    const real &dt,
    const real &rainRate,
    ADEsolver &ade,
    const real &dzb){
    #if SERGHEI_DEBUG_SEDIMENT>1
      std::cout << GGD << __PRETTY_FUNCTION__ << std::endl;
    #endif
    //----------------------------------------------------- 
    real dsi, fbedi, wsi, sedCharVeli, alphaEi, alphaDi;
    real qscap;
    real qs;
    real aux;
    real sedVol;

    real Drain=0.0;
    real Irain; //rainfall intensity (mm/h)
    real KE=0.0; //rainfall energy at ground (J/mm/m2)

    real wp2 = 0.0025; //square of water penetration length (we set it to 0.05 m)
    real sqPH = 0.55; //sqrt of the mean plant height (we set it to 30cm for a mediterranean forest), low limit 14cm

    // CVM calculation ----------------------------------    
    real BRE;
    real Cfactor;

    real a=16.;
    real b=6.789759633; //BRE(xz=-1)=0.9999
    real xz;
    if(ZBthick(ii)>TOL6){
      xz=dzb/ZBthick(ii); //normalized eroded thickness
      BRE=1./(1.+exp(a*xz+b));
    }else{ //null erodible layer thickness
      BRE=1.;
    }

    Cfactor = (1.-BRE)/exp(LUcoeff(ii)*FVCover(ii));
    if(Cfactor<TOL12) Cfactor=0.0;
    //--------------------------------------------------

    //available erodible thickness
    real tz;
    tz = ZBthick(ii) + dzb; 
    if(tz<0.0) tz=0.0;


    #if SERGHEI_RAIN_DETACHMENT
    if(h > z0){ 
      Irain = rainRate*3.6e6; //mm/h
      if(Irain >= 0.35){
        KE += FVCover(ii) * (15.8*sqPH - 5.87);
        KE += (1.-FVCover(ii)) * (8.95 + 8.44*mylog(Irain));
        Drain = BRE * kinterrill/RHOS * KE*Irain/3600. * exp(-h*h/wp2);
      }
    } 
    #endif      

    for(int iphi=iphised; iphi<(iphised+nSed); iphi++){
      dsi=ds(iphi-iphised);
      fbedi=fbed(iphi-iphised);
      wsi=ws(iphi-iphised);

      if(h > z0){  
        sedVol=0.0;

        //Rainfall-driven detachment
        #if SERGHEI_RAIN_DETACHMENT    
          sedVol += fbedi * Drain * dt; 
        #endif    

        //Runoff-driven detachment
        qscap=0.0;
        if(tau > TOL12){ //h>z0 && umod>0
          qscap = 0.01/((RHOS-RHOW)*GRAV) * umod*umod/wsi * tau; 

          //limiting qscap by flow discharge
          aux=fbedi*bedConc*qflow;
          if(qscap > aux) qscap = aux; 

          //Unit capacity
          qscap /=qflow;

          sedVol += Krill * Cfactor * fbedi * wsi * qscap * dt;
        }

        // erodible layer thikness limitation
        if(sedVol > fbedi*bedConc*tz){
          sedVol = fbedi*bedConc*tz;
        }

        ade.hphiSource(ii,iphi) += sedVol/dt;  

      }
      
      //Deposit
      qs=ade.hphi(ii,iphi)/h; //qs/qflow
      sedVol = BTrill * wsi * qs * dt; //use current dt for limiting, which can be different to real dt
      if(sedVol > ade.hphi(ii,iphi)){ //limit deposition
        sedVol = ade.hphi(ii,iphi);
      }
      ade.hphiSource(ii,iphi) -= sedVol/dt;      
      
    } 

  };
  #endif  


  #if SERGHEI_SUSPENDED_SEDIMENT
  KOKKOS_INLINE_FUNCTION void updateAndReset(const int ii, 
    const Domain &dom, 
    real &z, 
    real &h, 
    real &hu, 
    real &hv, 
    ADEsolver &ade,
    const real &hmin,
    real &phiTotal){
    #if SERGHEI_DEBUG_SEDIMENT>1
      std::cout << GGD << __PRETTY_FUNCTION__ << std::endl;
    #endif
    //-----------------------------------------------------
    real z0=0.0;
    bedExchangeVol(ii) = 0.; 
    real sedVol = 0.;
    real bedVol = 0.;
    real aux = 0.;
    real reducCoef = 0.;
    real phiflow = 0.;
    real rho = 0.;
    real hphi_total=0.0;

    //z0=fmax(dsmax,hmin); 
    //if(h > z0){
    if(h > TOL12){     
 
      for(int iphi=iphised; iphi<(iphised+nSed); iphi++){
        //compute bulk concentration
        phiflow += ade.hphi(ii,iphi)/h;

        //compute exchange term
        sedVol = ade.hphiSource(ii,iphi) * dom.dt;
        if(sedVol<0.0){ //limit deposition
          sedVol = fmax(-ade.hphi(ii,iphi),sedVol);
        }

        //storing sedVol
        ade.hphiSource(ii,iphi)=sedVol;

        //accumulate bulk exchange vol
        bedVol += sedVol/bedConc;
      }

      //bulk reduction coefficient
      rho=RHOW+(RHOS-RHOW)*phiflow;
      if(fabs(bedVol)>=TOL12){
        aux=bedVol;
        if(bedVol<0.0){
          aux=fmax(-h,aux);
          aux=fmin(0.0,aux); 
        }
        #if MOMENTUM_DISSIPATION
        else{ 
          aux=fmin(h*rho/(rhob-rho),aux);
          aux=fmax(0.0,aux);
        }
        #endif
        reducCoef=aux/bedVol;
      }

      //sediment exchange
      for(int iphi=iphised; iphi<(iphised+nSed); iphi++){
        ade.hphi(ii,iphi) += reducCoef*ade.hphiSource(ii,iphi);
        if(ade.hphi(ii,iphi)<TOL12){ 
          ade.hphi(ii,iphi)=0.0;
        }

        //bulk concentratron
        hphi_total += ade.hphi(ii,iphi);
      }
      
      //bed change
      #if SERGHEI_UPWIND_BED==0
      z -= reducCoef*bedVol;
      #endif

      //momentum dissipation
      #if MOMENTUM_DISSIPATION
      aux = 1. - momReductionFactor * (rhob-rho)/rho/h * reducCoef*bedVol;
      hu *= aux;
      hv *= aux;        
      #endif

      //flow change
      h += reducCoef*bedVol;
      if(h<TOL12){
        h=0.0;
        hu=0.0;
        hv=0.0;
        for(int iphi=iphised; iphi<(iphised+nSed); iphi++){
          ade.hphi(ii,iphi)=0.0;
        }        
      }

      //account for bed exchage
      bedExchangeVol(ii) = reducCoef*bedVol;

      //update phiTotal value
      phiTotal = hphi_total/h;

    }

  };  
  #endif 


  /////////////////////
  KOKKOS_INLINE_FUNCTION void upwinding(int id1, int id2, real z1, real z2, 
    real const &dt, real const &dx, 
    ADEsolver &ade){

    int id;
    real S0, deltaZk, deltaQsk;
    real lambdab=0.0;
    real NbL=0.0;
    real NbR=0.0;
    real deltaqL=0.0;
    real deltaqR=0.0;
    real sedVol = 0.0;

    for(int iphi=iphised; iphi<(iphised+nSed); iphi++){
      sedVol = ade.hphiSource(id1,iphi) * dt;
      if(sedVol<(-ade.hphi(id1,iphi))){ //limit deposition
        sedVol = -ade.hphi(id1,iphi);
      }
      NbL += sedVol/dt;

      sedVol = ade.hphiSource(id2,iphi) * dt;
      if(sedVol<(-ade.hphi(id2,iphi))){ //limit deposition
        sedVol = -ade.hphi(id2,iphi);
      }
      NbR += sedVol/dt;  
    }
    deltaQsk = 0.5*(NbL+NbR)*0.5*dx/bedConc;        
    deltaZk = z2-z1;

    S0=deltaZk/dx;
    if(fabs(S0)>TOLSB1){
      lambdab = deltaQsk/deltaZk;
    }

    if(lambdab>0.0){
        deltaqL = 0.0;
        deltaqR = deltaQsk;
    }else if(lambdab<0.0){
        deltaqL = deltaQsk;
        deltaqR = 0.0;
    }else{ //lambdab=0.0
        deltaqL = 0.25*NbL*dx/bedConc;
        deltaqR = 0.25*NbR*dx/bedConc;
    }    

    //// Geomorphological collapse ////////////////////////////////
    real zL, zR;
    real qstar=0.0;
    real dZstar;
    real dZmax;
    real reducCoef=0.5;

    dZmax = dx*tanDeltaf;

    zL = z1 - dt*deltaqL/dx;
    zR = z2 - dt*deltaqR/dx;
    dZstar=zR-zL;
    if(dZstar > dZmax){
      qstar=reducCoef*0.5*dx*(dZmax-dZstar)/dt;
    }else if(dZstar < -dZmax){
      qstar=-reducCoef*0.5*dx*(dZmax+dZstar)/dt;
    }
    //////////////////////////////////////////////////////////////

    dz0(id1) += deltaqL + qstar;
    dz1(id2) += deltaqR - qstar;    

  } 


  /////////////////////
  KOKKOS_INLINE_FUNCTION real getLambdaBcell(int id1, real z1, real z2, 
    real const &dx, ADEsolver &ade){

    int id;
    real S0, deltaZk, deltaQsk;
    real lambdab=0.0;
    real Nb=0.0;

    for(int iphi=iphised; iphi<(iphised+nSed); iphi++){ 
      Nb += ade.hphiSource(id1,iphi);
    }
    deltaQsk = Nb*0.5*dx/bedConc;        
    deltaZk = z2-z1;

    S0=deltaZk/dx;
    if(fabs(S0)>TOLSB2){
      lambdab = deltaQsk/deltaZk;
    }

    return(lambdab);
    
  
  }   


};

#endif
#endif
