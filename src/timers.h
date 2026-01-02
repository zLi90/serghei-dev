#pragma once

#include <fstream>
#include <iostream>
#include "Parallel.h"
#include <mpi.h>
class SergheiTimers {
    struct init_t {
        static constexpr unsigned short n = 1;
        double total = 0.0;
    };

    struct io_t {
        static constexpr unsigned short n = 3;
        double in = 0.0;
        double out = 0.0;
        double mpi = 0.0;
    };

    struct ss_t {
        static constexpr unsigned short n = 1;
        double raininf = 0.0;
    };

    struct flux_t {
        static constexpr unsigned short n = 3;
        double total = 0.0;
        double dt = 0.0;
        double mpi = 0.0;
    };

    struct update_t {
        static constexpr unsigned short n = 1;
        double total = 0.0;
    };
    
    struct wdc_t {
        static constexpr unsigned short n = 1;
        double total = 0.0;
    };

    struct halo_t {
        static constexpr unsigned short n = 2;
        double total = 0.0;
        double mpi = 0.0;
    };

    struct bc_t {
        static constexpr unsigned short n = 3;
        double total = 0.0;
        double integrate = 0.0;
        double mpi = 0.0;
    };

    struct dt_t {
        static constexpr unsigned short n = 2;
        double total = 0.0;
        double mpi = 0.0;
    };

    struct integrate_t {
        static constexpr unsigned short n = 2;
        double total = 0.0;
        double mpi = 0.0;
    };

    struct swe_t {
        // WARNING: make sure to account for the sizes of the nested structs
        static constexpr unsigned short n = 4 + init_t::n + io_t::n + ss_t::n + 
                                         flux_t::n + update_t::n + wdc_t::n + halo_t::n + 
                                         integrate_t::n + bc_t::n + dt_t::n;
        double total = 0.0;
        double solve = 0.0;
        double other = 0.0;
        double mpi = 0.0;
        init_t init;
        io_t io;
        flux_t flux;
        update_t update;
        wdc_t wetdrycorr;
        halo_t halo;
        dt_t dt;
        bc_t bc;
        ss_t ss;
        integrate_t integrate;
    };

    #if SERGHEI_LPT || SERGHEI_LPT_RK || SERGHEI_LPT_RK_OFFLINE
    struct lpt_t {
        static constexpr unsigned short n = 3;
        double update = 0;
        double init   = 0;
        double out    = 0;
    };
    #endif

    #if SERGHEI_RE_MODEL
    struct re_t {
        static constexpr unsigned short n = 10;
        double gw           = 0;
        double gwBC         = 0;
        double gwlinsys     = 0;
        double gwlinsol     = 0;
        double gwUpdateK    = 0;
        double gwUpdateQ    = 0;
        double gwUpdateWC   = 0;
        double gwMPI        = 0;
        double gwIntegrate  = 0;
        double out          = 0;     
    };
    #endif

public:
    double total = 0.0;
    swe_t swe;
    #if SERGHEI_LPT || SERGHEI_LPT_RK || SERGHEI_LPT_RK_OFFLINE
    lpt_t lpt;
    #endif
    #if SERGHEI_RE_MODEL
    re_t re;
    #endif

private:
    swe_t* g_swe = nullptr;

    #if SERGHEI_LPT || SERGHEI_LPT_RK || SERGHEI_LPT_RK_OFFLINE
    lpt_t* g_lpt = nullptr;
    #endif

    #if SERGHEI_RE_MODEL
    re_t* g_re = nullptr;
    #endif

public:
    SergheiTimers() = default;

    void closure() {
        swe.other = swe.solve - (swe.flux.total + swe.update.total + swe.wetdrycorr.total + swe.dt.total + 
                  swe.bc.total + swe.halo.total + swe.ss.raininf);
        swe.total = swe.solve + swe.init.total + swe.io.in + swe.io.out + 
                    swe.integrate.total;
        swe.mpi = swe.io.mpi + swe.flux.mpi + swe.halo.mpi + swe.dt.mpi + swe.bc.mpi + swe.integrate.mpi;
    }

    void gather(Parallel const &par) {
        if(par.masterproc) g_swe = new swe_t[par.nranks];

        // Gather all timer values as doubles
        MPI_Gather(
            reinterpret_cast<double*>(&swe),
            swe_t::n,
            MPI_DOUBLE,
            par.masterproc ? reinterpret_cast<double*>(g_swe) : nullptr,
            swe_t::n,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );

        #if SERGHEI_LPT || SERGHEI_LPT_RK || SERGHEI_LPT_RK_OFFLINE
        if(par.masterproc) g_lpt = new lpt_t[par.nranks];
        MPI_Gather(&lpt, lpt_t::n, MPI_DOUBLE,
                   par.masterproc ? g_lpt : nullptr, lpt_t::n, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
        #endif

        #if SERGHEI_RE_MODEL
        if(par.masterproc) g_re = new re_t[par.nranks];
        MPI_Gather(&re, re_t::n, MPI_DOUBLE,
                   par.masterproc ? g_re : nullptr, re_t::n, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
        #endif
    }

    void computeRelative(const SergheiTimers &base){
        total /= base.total;
        swe.solve /=  base.total;
        swe.init.total /= base.total;
        swe.io.in /= base.total;
        swe.io.out /= base.total; 
        swe.io.mpi /= base.total; 
        swe.flux.total /= base.total;
        swe.flux.dt /= base.total;
        swe.flux.mpi /= base.total;
        swe.update.total /= base.total;
        swe.wetdrycorr.total /= base.total;
        swe.halo.total /= base.total;
        swe.halo.mpi /= base.total;
        swe.dt.total /= base.total;
        swe.dt.mpi /= base.total;
        swe.bc.total /= base.total;
        swe.bc.integrate /= base.total;
        swe.bc.mpi /= base.total;
        swe.ss.raininf /= base.total;
        swe.integrate.total /= base.total;
        swe.integrate.mpi /= base.total; 
        swe.mpi /= base.total;

        #if SERGHEI_LPT || SERGHEI_LPT_RK || SERGHEI_LPT_RK_OFFLINE
        lpt.update /= base.total;
        lpt.init /= base.total;
        lpt.out /= base.total;
        #endif

        #if SERGHEI_RE_MODEL
        re.gw /= base.total;
        re.gwBC /= base.total;
        re.gwlinsys /= base.total;
        re.gwlinsol /= base.total;
        re.gwUpdateK /= base.total;
        re.gwUpdateQ /= base.total;
        re.gwUpdateWC /= base.total;
        re.gwMPI /= base.total;
        re.gwIntegrate /= base.total;
        re.out /= base.total;
        #endif
    }

    inline std::string indent(std::size_t n){
        return std::string(n, ' ');
    }
    inline std::string level(std::size_t n){
        return indent(2*n);
    }   

    void report(std::ofstream &file, int nrank){

        file << "serghei.total";
        for(int r = 0; r < nrank; ++r) file << "\t" << total;
        file << std::endl;

        file << level(1) << "swe.total";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].total;
        file << std::endl;

        file << level(2) << "swe.init.total"; 
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].init.total;
        file << std::endl;

        file << level(2) <<"swe.io.in";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].io.in;
        file << std::endl;

        file << level(2) << "swe.io.out";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].io.out;
        file << std::endl;
        
        file << level(2) << "swe.io.mpi";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].io.mpi;
        file << std::endl;
        
        file << level(2) << "swe.solve";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].solve;
        file << std::endl;

        file << level(3) << "swe.flux.total";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].flux.total;
        file << std::endl;
        
        file << level(4) << "swe.flux.dt";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].flux.dt;
        file << std::endl;

        file << level(4) << "swe.flux.mpi";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].flux.mpi;
        file << std::endl;

        file << level(3) << "swe.update.total";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].update.total;
        file << std::endl;

        file << level(3) << "swe.wetdrycorr.total";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].wetdrycorr.total;
        file << std::endl;

        file << level(3) << "swe.halo.total";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].halo.total;
        file << std::endl;

        file << level(4) << "swe.halo.mpi";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].halo.mpi;
        file << std::endl;

        file << level(3) << "swe.dt.total";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].dt.total;
        file << std::endl;

        file << level(4) << "swe.dt.mpi";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].dt.mpi;
        file << std::endl;

        file << level(3) << "swe.bc.total";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].bc.total;
        file << std::endl;

        file << level(4) << "swe.bc.integrate";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].bc.integrate;
        file << std::endl;

        file << level(4) << "swe.bc.mpi";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].bc.mpi;
        file << std::endl;

        file << level(3) << "swe.ss.raininf";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].ss.raininf;
        file << std::endl;

        file << level(3) << "swe.other";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].other;
        file << std::endl;

        file << level(2) << "swe.integrate.total";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].integrate.total;
        file << std::endl;

        file << level(3) << "swe.integrate.mpi";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].integrate.mpi;
        file << std::endl;
        
        file << level(2) << "(swe.mpi)";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].mpi;
        file << std::endl;


        #if SERGHEI_LPT || SERGHEI_LPT_RK || SERGHEI_LPT_RK_OFFLINE
        // Report LPT timers
        file << level(1) << "lpt.update";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_lpt[r].update;
        file << std::endl;

        file << level(1) << "lpt.init";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_lpt[r].init;
        file << std::endl;

        file << level(1) << "lpt.out";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_lpt[r].out;
        file << std::endl;
        #endif

        #if SERGHEI_RE_MODEL
        file << level(1) << "re.gw";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_re[r].gw;
        file << std::endl;

        file << level(2) << "re.gwBC";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_re[r].gwBC;
        file << std::endl;

        file << level(2) << "re.gwlinsys";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_re[r].gwlinsys;
        file << std::endl;

        file << level(2) << "re.gwlinsol";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_re[r].gwlinsol;
        file << std::endl;

        file << level(2) << "re.gwUpdateK";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_re[r].gwUpdateK;
        file << std::endl;

        file << level(2) << "re.gwUpdateQ";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_re[r].gwUpdateQ;
        file << std::endl;

        file << level(2) << "re.gwUpdateWC";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_re[r].gwUpdateWC;
        file << std::endl;

        file << level(2) << "re.gwMPI";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_re[r].gwMPI;
        file << std::endl;

        file << level(2) << "re.gwIntegrate";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_re[r].gwIntegrate;
        file << std::endl;

        file << level(1) << "re.out";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_re[r].out;
        file << std::endl;
        #endif
    }

    double inline getAverageTimer(double mytimer, int nrank){
        double avg = 0.0;
        for(int r=0; r<nrank; ++r) avg += mytimer;
        if(avg == 0.0) return 1.0;
        return avg / static_cast<double>(nrank);
    }

    double inline getAverageTimer(std::function<double(int)> getTimer, int nrank){
        double avg = 0.0;
        for(int r = 0; r < nrank; ++r) avg += getTimer(r);
        if(avg == 0.0) return 1.0;
        return avg / static_cast<double>(nrank);
    }

    void reportLoadBalance(std::ofstream &file, int nrank){
        double avg = 0.0;

        avg = getAverageTimer([this](int r){ return g_swe[r].total; }, nrank);
        file << "serghei.total";
        for(int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].total / avg;
        file << std::endl;

        avg = getAverageTimer([this](int r){ return g_swe[r].total; }, nrank);
        file << level(1) << "swe.total";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].total / avg;
        file << std::endl;

        avg = getAverageTimer([this](int r){ return g_swe[r].init.total; }, nrank);
        file << level(2) << "swe.init.total"; 
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].init.total / avg;
        file << std::endl;

        avg = getAverageTimer([this](int r){ return g_swe[r].io.in; }, nrank);
        file << level(2) <<"swe.io.in";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].io.in / avg;
        file << std::endl;

        avg = getAverageTimer([this](int r){ return g_swe[r].io.out; }, nrank);
        file << level(2) << "swe.io.out";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].io.out / avg;
        file << std::endl;
        
        avg = getAverageTimer([this](int r){ return g_swe[r].io.mpi; }, nrank);
        file << level(2) << "swe.io.mpi";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].io.mpi / avg;
        file << std::endl;
        
        avg = getAverageTimer([this](int r){ return g_swe[r].solve; }, nrank);
        file << level(2) << "swe.solve";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].solve / avg;
        file << std::endl;

        avg = getAverageTimer([this](int r){ return g_swe[r].flux.total; }, nrank);
        file << level(3) << "swe.flux.total";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].flux.total / avg;
        file << std::endl;
        
        avg = getAverageTimer([this](int r){ return g_swe[r].flux.dt; }, nrank);
        file << level(4) << "swe.flux.dt";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].flux.dt / avg;
        file << std::endl;

        avg = getAverageTimer([this](int r){ return g_swe[r].flux.mpi; }, nrank);
        file << level(4) << "swe.flux.mpi";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].flux.mpi / avg;
        file << std::endl;

        avg = getAverageTimer([this](int r){ return g_swe[r].update.total; }, nrank);
        file << level(3) << "swe.update.total";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].update.total / avg;
        file << std::endl;

        avg = getAverageTimer([this](int r){ return g_swe[r].wetdrycorr.total; }, nrank);
        file << level(3) << "swe.wetdrycorr.total";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].wetdrycorr.total / avg;
        file << std::endl;

        avg = getAverageTimer([this](int r){ return g_swe[r].halo.total; }, nrank);
        file << level(3) << "swe.halo.total";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].halo.total / avg;
        file << std::endl;

        avg = getAverageTimer([this](int r){ return g_swe[r].halo.mpi; }, nrank);
        file << level(4) << "swe.halo.mpi";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].halo.mpi / avg;
        file << std::endl;

        avg = getAverageTimer([this](int r){ return g_swe[r].dt.total; }, nrank);
        file << level(3) << "swe.dt.total";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].dt.total / avg;
        file << std::endl;

        avg = getAverageTimer([this](int r){ return g_swe[r].dt.mpi; }, nrank);
        file << level(4) << "swe.dt.mpi";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].dt.mpi / avg;
        file << std::endl;

        avg = getAverageTimer([this](int r){ return g_swe[r].bc.total; }, nrank);
        file << level(3) << "swe.bc.total";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].bc.total / avg;
        file << std::endl;

        avg = getAverageTimer([this](int r){ return g_swe[r].bc.integrate; }, nrank);
        file << level(4) << "swe.bc.integrate";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].bc.integrate / avg;
        file << std::endl;

        avg = getAverageTimer([this](int r){ return g_swe[r].bc.mpi; }, nrank);
        file << level(4) << "swe.bc.mpi";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].bc.mpi / avg;
        file << std::endl;

        avg = getAverageTimer([this](int r){ return g_swe[r].ss.raininf; }, nrank);
        file << level(3) << "swe.ss.raininf";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].ss.raininf / avg;
        file << std::endl;

        avg = getAverageTimer([this](int r){ return g_swe[r].other; }, nrank);
        file << level(3) << "swe.other";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].other / avg;
        file << std::endl;

        avg = getAverageTimer([this](int r){ return g_swe[r].integrate.total; }, nrank);
        file << level(2) << "swe.integrate.total";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].integrate.total / avg;
        file << std::endl;

        avg = getAverageTimer([this](int r){ return g_swe[r].integrate.mpi; }, nrank);
        file << level(3) << "swe.integrate.mpi";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].integrate.mpi / avg;
        file << std::endl;
        
        avg = getAverageTimer([this](int r){ return g_swe[r].mpi; }, nrank);
        file << level(2) << "(swe.mpi)";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_swe[r].mpi / avg;
        file << std::endl;

        #if SERGHEI_LPT || SERGHEI_LPT_RK || SERGHEI_LPT_RK_OFFLINE
        avg = getAverageTimer([this](int r){ return g_lpt[r].update; }, nrank);
        file << level(1) << "lpt.update";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_lpt[r].update / avg;
        file << std::endl;

        avg = getAverageTimer([this](int r){ return g_lpt[r].init; }, nrank);
        file << level(1) << "lpt.init";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_lpt[r].init / avg;
        file << std::endl;

        avg = getAverageTimer([this](int r){ return g_lpt[r].out; }, nrank);
        file << level(1) << "lpt.out";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_lpt[r].out / avg;
        file << std::endl;
        #endif

        #if SERGHEI_RE_MODEL
        avg = getAverageTimer([this](int r){ return g_re[r].gw; }, nrank);
        file << level(1) << "re.gw";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_re[r].gw / avg;
        file << std::endl;

        avg = getAverageTimer([this](int r){ return g_re[r].gwBC; }, nrank);
        file << level(2) << "re.gwBC";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_re[r].gwBC / avg;
        file << std::endl;

        avg = getAverageTimer([this](int r){ return g_re[r].gwlinsys; }, nrank);
        file << level(2) << "re.gwlinsys";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_re[r].gwlinsys / avg;
        file << std::endl;

        avg = getAverageTimer([this](int r){ return g_re[r].gwlinsol; }, nrank);
        file << level(2) << "re.gwlinsol";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_re[r].gwlinsol / avg;
        file << std::endl;

        avg = getAverageTimer([this](int r){ return g_re[r].gwUpdateK; }, nrank);
        file << level(2) << "re.gwUpdateK";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_re[r].gwUpdateK / avg;
        file << std::endl;

        avg = getAverageTimer([this](int r){ return g_re[r].gwUpdateQ; }, nrank);
        file << level(2) << "re.gwUpdateQ";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_re[r].gwUpdateQ / avg;
        file << std::endl;

        avg = getAverageTimer([this](int r){ return g_re[r].gwUpdateWC; }, nrank);
        file << level(2) << "re.gwUpdateWC";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_re[r].gwUpdateWC / avg;
        file << std::endl;

        avg = getAverageTimer([this](int r){ return g_re[r].gwMPI; }, nrank);
        file << level(2) << "re.gwMPI";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_re[r].gwMPI / avg;
        file << std::endl;

        avg = getAverageTimer([this](int r){ return g_re[r].gwIntegrate; }, nrank);
        file << level(2) << "re.gwIntegrate";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_re[r].gwIntegrate / avg;
        file << std::endl;

        avg = getAverageTimer([this](int r){ return g_re[r].out; }, nrank);
        file << level(1) << "re.out";
        for (int r = 0; r < nrank; ++r) file << "\t" << g_re[r].out / avg;
        file << std::endl;
        #endif
    }
};
