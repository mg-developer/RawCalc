//////////////////////////////////////////////
//Export as library
#define RawCalc_EXPORTS
#include "Exports.h"
/////////////////////////////////////////////

#include <fstream>
#include <memory>
#include <string.h>
#include "Logging.h"
#include "RawStructs.h"
#include "ProgGPU.h"


int calcOnGPU(const ProcessParams &p);
int findBestGPU();

MDB &to16(const metadata &, const MDB &);
MDB &to16H(const metadata &, const MDB &);
MDB &to12(const metadata &, const MDB &);
MDB &from12to16(const metadata &, const MDB &);


//Global Data
static MDB inputData;
static std::string inputFile;
static metadata metaData;

static int ReadData(const char *file, const mlvBlock &block);
static int WriteData(const char *file, const MDB &processed, const unsigned char *dng, int sizedng);

template<typename T>
int write_raw(const char *name, const T& in, std::size_t size = sizeof(T)) {
    std::ofstream f(name, std::ofstream::binary);
    f.write((const char*)&in, size);
    f.close();
    return 0;
}


extern "C" {

RawCalc_EXPORT int ModuleSetup(metadata data) {
    LOG() << "ModuleSetup" << std::endl;

    memcpy((void*)&metaData, (void*)&data, sizeof(metadata));
    inputData.reset(metaData.stripByteCountReal);

    //Reset settings
    findBestGPU();

    LOG() << "Res:" << data.xResolution << "x" << data.yResolution << std::endl;
    return 0;
}

RawCalc_EXPORT int AssignFileToProcess(char *input) {
//    LOG() << "Input file:" << input << std::endl;
    inputFile = std::move(std::string(input));
    return 0;
}


RawCalc_EXPORT int ProcessDataAndSave(char *output, mlvBlock block, unsigned char *dngheader, int sizedng) {
    LOG() << "Output file:" << output << "  Frame:" << block.MLVFrameNo << std::endl;

    ReadData(inputFile.c_str(), block);

    MDB &processed = to16H(metaData, inputData);
//   MDB &processed = to16(metaData, inputData);

    //TODO:
    //vertical Banding
    //chromaSmoothing
    //pinkHighlight
    //downsample to 12

    WriteData(output, processed, dngheader, sizedng);

    return 0;
}

}//extern "C"

static int ReadData(const char *file, const mlvBlock &block) {
    std::ifstream f(file, std::ifstream::binary);
    if(f.is_open()) {

        long long pos = (long long)(block.fileOffset + 32 + block.EDMACoffset);
        f.seekg(pos, f.beg);
        char *p = (char *)inputData();
        f.read(p, inputData.bytes());
        f.close();
    }
    return 0;
}

static int WriteData(const char *file,const MDB &processed,const unsigned char *dng, int sizedng) {
    std::ofstream f(file, std::ofstream::binary);
    if(f.is_open()) {
        f.write((const char*)dng, sizedng);
        f.write((const char*)processed(), processed.bytes());
        f.write((const char*)metaData.versionString, 128);
        f.close();
    }
    return 0;
}


