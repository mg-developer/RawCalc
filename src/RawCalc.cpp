#include "Logging.h"
#include "Exports.h"
#include "ProgGPU.h"

extern "C" int calcOnGPU(const ProcessParams &p);


RawCalc_EXPORT int ModuleSetup() {

    LOG() << "ModuleSetup";

    ProcessParams p;
    p.data = ManagedPtr<int>(100);

    for(int n=0; n<100; ++n)
        p.data[n] = n;

    calcOnGPU(p);

    return 0;
}



