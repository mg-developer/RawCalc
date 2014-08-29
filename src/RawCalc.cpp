//////////////////////////////////////////////
//Export as library
#define RawCalc_EXPORTS
#include "Exports.h"
/////////////////////////////////////////////


#include "Logging.h"
#include "ProgGPU.h"

int calcOnGPU(const ProcessParams &p);
int findBestGPU();

extern "C" {

RawCalc_EXPORT int ModuleSetup() {

    LOG() << "ModuleSetup" << std::endl;

    findBestGPU();

    ProcessParams p;
    p.data = ManagedPtr<int>(100);

    for(int n=0; n<100; ++n)
        p.data[n] = n;

    calcOnGPU(p);

    LOG() << "Finished" << std::endl;
    return 0;
}

}

