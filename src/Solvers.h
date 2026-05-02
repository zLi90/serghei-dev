#ifndef _SOLVERS_H_
#define _SOLVERS_H_

#include "define.h"
#include "SArray.h"

#ifndef SERGHEI_REBALANCE_SOLVER_CONTRIBUTIONS
#define SERGHEI_REBALANCE_SOLVER_CONTRIBUTIONS 0
#endif

class Solver{
  public:
  
	mutable SArray<real,3> upwP, upwM;
	mutable real utilde=0, vtilde=0, htilde=0, ctilde=0, frictionSlope=0;
	mutable real numFlux=0; //numerical (normal) mass flux


KOKKOS_INLINE_FUNCTION void roe(const SArray<real,5> &s1, const SArray<real,5> &s2, real const &dt, real const &dx, real const &nx, real const &ny) const {

	SArray<real,3> lambda, lambdaE, alpha, beta, diff;
	SArray<real,3,3> eigenV;
	real e1, e2, u1, u2, v1, v2;
	real un;
	int k;

	real betaB=0.0;
	real betaF=0.0;

	real h1=s1(idH);
	real h2=s2(idH);
	real hu1=s1(idHU);
	real hu2=s2(idHU);
	real hv1=s1(idHV);
	real hv2=s2(idHV);
	real z1=s1(idZ);
	real z2=s2(idZ);
	real n1=s1(idR);
	real n2=s2(idR);


	real sqrt1=sqrt(h1);
	real sqrt2=sqrt(h2);

	if(h1>0.0){
		u1=hu1/h1;
		v1=hv1/h1;
	}else{
		u1=0.0;
		v1=0.0;
	}

	if(h2>0.0){
		u2=hu2/h2;
		v2=hv2/h2;
	}else{
		u2=0.0;
		v2=0.0;
	}

	// Compute interface values
	htilde = 0.5 * ( h1+h2 );
	utilde = ( u1*sqrt1 + u2*sqrt2) / (sqrt1 + sqrt2);
	vtilde = ( v1*sqrt1 + v2*sqrt2) / (sqrt1 + sqrt2);	
	ctilde = mysqrt(GRAV*htilde);

	un=utilde*nx+vtilde*ny;

	lambda(0)=un-ctilde;
	lambda(1)=un;
	lambda(2)=un+ctilde;

	//entropy correction
	lambdaE(0)=0.0;
	lambdaE(1)=0.0;
	lambdaE(2)=0.0;

	e1=u1*nx+v1*ny-SQRTGRAV*sqrt1;
	e2=u2*nx+v2*ny-SQRTGRAV*sqrt2;

	if(e1<0.0 && e2>0.0){
		lambdaE(0)=lambda(0) - e1*(e2-lambda(0))/(e2-e1);
		lambda(0)=e1*(e2-lambda(0))/(e2-e1);
	}

	e1=u1*nx+v1*ny+SQRTGRAV*sqrt1;
	e2=u2*nx+v2*ny+SQRTGRAV*sqrt2;

	if(e1<0.0 && e2>0.0){
		lambdaE(2)=lambda(2) - e2*(lambda(2)-e1)/(e2-e1);
		lambda(2)=e2*(lambda(2)-e1)/(e2-e1);
	}

	eigenV(0,0)=1.0;
	eigenV(0,1)=utilde-ctilde*nx;
	eigenV(0,2)=vtilde-ctilde*ny;
	eigenV(1,0)=0.0;
	eigenV(1,1)=-ctilde*ny;
	eigenV(1,2)=ctilde*nx;
	eigenV(2,0)=1.0;
	eigenV(2,1)=utilde+ctilde*nx;
	eigenV(2,2)=vtilde+ctilde*ny;

	diff(0)=h2-h1;
	diff(1)=hu2-hu1;
	diff(2)=hv2-hv1;

	alpha(0)=0.5*(diff(0)-((diff(1)*nx+diff(2)*ny) - un*diff(0))/ctilde);
	alpha(1)=((diff(2)-vtilde*diff(0))*nx - (diff(1)-utilde*diff(0))*ny)/ctilde;
	alpha(2)=0.5*(diff(0)+((diff(1)*nx+diff(2)*ny) - un*diff(0))/ctilde);

	real deltaz=z2-z1;
	real l1=z1+h1;
	real l2=z2+h2;
	real dzp=deltaz;
	real hp;

	if(deltaz>=0.0){
		hp=h1;
		if(l1<z2){
			dzp=h1;
		}
	}else{
		hp=h2;
		if(l2<z1){
			dzp=-h2;
		}
	}

	betaB=0.5/ctilde*GRAV*(hp-0.5*fabs(dzp))*dzp;

	#if SERGHEI_POINTWISE_FRICTION==0
		// interface value of roughness
		real n = 0.5*(n1+n2);

		real gamma, hbeta;
		real frictionSlope = 0.0;

		#if SERGHEI_FRICTION_MODEL == SERGHEI_FRICTION_MANNING
			gamma = n*n;
			hbeta = htilde*cbrt(htilde);
		#endif
		#if SERGHEI_FRICTION_MODEL == SERGHEI_FRICTION_DARCYWEISBACH
			// n represents friction factor f
			gamma = n/(8*GRAV);
			hbeta = htilde;
		#endif
		#if SERGHEI_FRICTION_MODEL == SERGHEI_FRICTION_CHEZY
			// n represents chezy roughness C
			gamma = 1./(n*n);
			hbeta = htilde;
		#endif

		frictionSlope =  un*sqrt(utilde*utilde+vtilde*vtilde)*gamma/hbeta;

		//betaF=0.5*c*n*n*un*sqrt(u*u+v*v)/(h*cbrt(h))*dx;
		betaF=0.5*ctilde*frictionSlope*dx;

		if(fabs(betaF)>TOLDRY){
		real qS=(hu1*nx+hv1*ny)*dx*0.5+fabs(lambda(0))*dt*(lambda(0)*alpha(0)-betaB);
		real qSF=qS+fabs(lambda(0))*dt*(-betaF);
		if(qS*qSF<0.0){
			betaF=(hu1*nx+hv1*ny)*dx*0.5/(fabs(lambda(0))*dt)+lambda(0)*alpha(0)-betaB;
		}
		qS=(hu2*nx+hv2*ny)*dx*0.5+fabs(lambda(2))*dt*(-lambda(2)*alpha(2)-betaB);
		qSF=qS+fabs(lambda(2))*dt*(-betaF);
		if(qS*qSF<0.0){
			betaF=(hu2*nx+hv2*ny)*dx*0.5/(fabs(lambda(2))*dt)-lambda(2)*alpha(2)-betaB;
		}
		}
	#endif

	beta(0)=betaB+betaF;


	//wet-wet subcritical correction
	hp = h1 + alpha(0);

	if (lambda(0) * lambda(2)<0.0 && hp>0.0 && h1 > 0.0 && h2 > 0.0){
		beta(0)=max(beta(0),alpha(0)*lambda(0)-h1*dx*0.5/dt);
		beta(0)=min(beta(0),-alpha(2)*lambda(2)+h2*dx*0.5/dt);
	}


	beta(1)=0.0;
	beta(2)=-beta(0);


	for(k=0;k<3;k++){
		upwM(k)=0.0;
		upwP(k)=0.0;
	}

	l1= hp - beta(0)/lambda(0); //left intermediate state
	l2= hp + beta(2)/lambda(2); //right intermediate state

	if(l2<-TOLDRY && h2 < TOLDRY){
		upwM(0)+=(lambda(0)*alpha(0) - beta(0)) + (lambda(2)*alpha(2) - beta(2));
	}else{
		if(l1<-TOLDRY && h1 < TOLDRY){
			upwP(0)+=(lambda(0)*alpha(0) - beta(0)) + (lambda(2)*alpha(2) - beta(2));
		}else{
			for(k=0;k<3;k++){ //regular contributions
				if(lambda(k)>0.0){
					upwP(0)+=(lambda(k)*alpha(k) - beta(k))*eigenV(k,0);
					upwP(1)+=(lambda(k)*alpha(k) - beta(k))*eigenV(k,1);
					upwP(2)+=(lambda(k)*alpha(k) - beta(k))*eigenV(k,2);
				}else{
					upwM(0)+=(lambda(k)*alpha(k) - beta(k))*eigenV(k,0);
					upwM(1)+=(lambda(k)*alpha(k) - beta(k))*eigenV(k,1);
					upwM(2)+=(lambda(k)*alpha(k) - beta(k))*eigenV(k,2);
				}
			}
		}

	}

	for(k=0;k<3;k++){ //entropy
		//entropy
		if(lambdaE(k)>0.0){
			upwP(0)+=(lambdaE(k)*alpha(k))*eigenV(k,0);
			upwP(1)+=(lambdaE(k)*alpha(k))*eigenV(k,1);
			upwP(2)+=(lambdaE(k)*alpha(k))*eigenV(k,2);
		}else{
			upwM(0)+=(lambdaE(k)*alpha(k))*eigenV(k,0);
			upwM(1)+=(lambdaE(k)*alpha(k))*eigenV(k,1);
			upwM(2)+=(lambdaE(k)*alpha(k))*eigenV(k,2);
		}
	}
	
	//Numerical Flux for the scalar transport
	numFlux=0.0;
	#if SERGHEI_REBALANCE_SOLVER_CONTRIBUTIONS
		numFlux= hu1 * nx + hv1 * ny + upwM(0);
		
		if(fabs(numFlux)<TOL14){
			numFlux=0.0;
			upwM(0) = - (hu1 * nx + hv1 * ny);
			upwP(0) =   (hu2 * nx + hv2 * ny);
		}

		for(k=1;k<3;k++){
			if(fabs(upwM(k))<TOL14) upwM(k)=0.0;
			if(fabs(upwP(k))<TOL14) upwP(k)=0.0;
		}
	#else
		for(k=0;k<3;k++){
			if(fabs(upwM(k))<TOL14) upwM(k)=0.0;
			if(fabs(upwP(k))<TOL14) upwP(k)=0.0;
		}
		numFlux= hu1 * nx + hv1 * ny + upwM(0);
	#endif

}

};
#endif
