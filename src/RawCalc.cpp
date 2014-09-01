//////////////////////////////////////////////
//Export as library
#define RawCalc_EXPORTS
#include "Exports.h"
/////////////////////////////////////////////

#include "Logging.h"
#include "RawStructs.h"
#include "ProgGPU.h"


int calcOnGPU(const ProcessParams &p);
int findBestGPU();

managed_data_block to16(datatype);
managed_data_block to16(datatype);
managed_data_block from12to16(datatype);


extern "C" {

RawCalc_EXPORT int ModuleSetup() {

    LOG() << "ModuleSetup" << std::endl;

    findBestGPU();

    ProcessParams p;
    p.data.reset(100);

    for(int n=0; n<100; ++n)
        p.data[n] = n;

    calcOnGPU(p);

    LOG() << "Finished" << std::endl;
    return 0;
}

RawCalc_EXPORT int AssignFileToProcess(char *input, metadata data) {
    LOG() << "Input file:" << input <<  "  Res:" << data.xResolution << "x" << data.yResolution << std::endl;
    LOG() << "Metadata size:" << sizeof(metadata) << std::endl;
    return 0;
}


RawCalc_EXPORT int ProcessDataAndSave(char *output, unsigned char *dngheader, int sizedng) {
    LOG() << "Output file:" << output << std::endl;
    for(int i=0; i<  sizedng; ++i) {
        LOG() << std::hex << dngheader[i] << " ";
    }
    LOG() << std::endl;
    return 0;
}

}
