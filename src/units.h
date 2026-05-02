#ifndef _UNITS_H_
#define _UNITS_H_

namespace Units{

  // this is to parse the time units following CF conventions
  std::string parseTimeUnits(std::string &in){
    // Find the position of the first occurrence of "since"
    size_t pos = in.find(" since");

    // If "since" is found, return the substring up to that point
    if (pos != std::string::npos) return in.substr(0, pos);

    // If "since" is not found, return the entire string
    return(in);
  }

  std::string parseTimeUnits(char* in){
    std::string stringin(in);
    return parseTimeUnits(stringin);
  }

  bool validateUnits(const std::string &var, const std::string &in, const std::string &expected){
    if(!in.compare(expected)) return(true);

    std::cerr << RERROR << "Expected units for " << GREEN << var << RESET << " is " << GREEN << expected << RESET << " but read " << RED << in  << RESET << std::endl;
    return(false);
  }

  real timeFactor(const std::string &in){
    if(!in.compare("d") || !in.compare("day") || !in.compare("days")) return(86400.0);  // day to seconds
    if(!in.compare("h") || !in.compare("hour") || !in.compare("hours")) return(3600.0);  // hours to seconds
    if(!in.compare("min") || !in.compare("minute") || !in.compare("minutes")) return(60.0);  // minutes to seconds
	  if(!in.compare("s") || !in.compare("second") || !in.compare("seconds")) return(1.0);
    std::cerr << RERROR << "Unhandled time units '" << in << "'. Failed conversion." << std::endl;
    abort();
  }

  real rainFactor(const std::string &in, const std::string &out){
    real ifactor, ofactor;
    bool ok=0;
    if(!in.compare("m/s")){
      ifactor = 1.0;
      ok=1;
    }
    if(!in.compare("mm/h")){
      ifactor = 0.001/3600.0; // mm/h to m/s
      ok=1;
    }
    if(!in.compare("mm/s")){
      ifactor = 0.001; // mm/h to m/s
      ok=1;
    }
    if(!ok){
      std::cerr << RERROR << "Unhandled source units '" << in << "'. Failed conversion." << std::endl;
      abort();
    }
    ok=0;
    if(!out.compare("mm/h")){
      ofactor = 3600*1000; // m/s to mm/h
      ok=1;
    }
    if(!out.compare("mm/s")){
      ofactor = 1000; // m/s to mm/s
      ok=1;
    }
    if(!out.compare("m/s")){
      ofactor = 1.0;
      ok=1;
    }
    if(!ok){
      std::cerr << RERROR << "Unhandeld target units " << out << " . Failed conversion." << std::endl;
      abort();
    }

    return(ifactor*ofactor);
  }

  real rainFactor(const std::string &in){
    return(rainFactor(in,"m/s"));
  }

  real rainFactor(const char *in){
  //  std::string s_in = std::string(in);
    return ( rainFactor(std::string(in)) );
  }
};
#endif
