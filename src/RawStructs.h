#ifndef _RAW_STRUCTS_H
#define _RAW_STRUCTS_H

#include <string>
#include <list>
#include "ManagedMem.h"

//#include "ArrayPtr.hpp"

enum quality { lowgrey = 0,
        lowmullim = 1,
        highgamma2 = 2,
        high709 =3
};

struct metadata;
struct audiodata;
struct filedata;
struct lensdata;
//struct threaddata;
struct convertSettings;
struct rawBlock;
struct mlvBlock;
struct Blocks;


#pragma pack(1)
struct metadata
{
    int xResolution;
    int yResolution;
    int frames;
    int bitsperSample;
    int bitsperSampleChanged;
    int isLog; //bool
    unsigned char colorMatrix[72];
    int lostFrames;
    int fpsNom;
    int fpsDen;
    int dropFrame;//bool
    unsigned char fpsString[16]; //temp - string
    int stripByteCount;
    int stripByteCountReal;
    unsigned char modell[64]; //temp - string
    unsigned char camId[32]; //temp - string
    int apiVersion;
    int splitCount;
    int photoRAW;//bool
    int  photoRAWe; //bool
    int RGGBValues[4];
    int RGBfraction[6];
    double jpgConvertR;
    double jpgConvertG;
    double jpgConvertB;
    int whiteBalance;
    int whiteBalanceMode;
    int whiteBalanceGM;
    int whiteBalanceBA;

    // variables for blackpoint and maximizing
    int blackLevelOld;
    int blackLevelNew;
    int whiteLevelOld;
    int whiteLevelNew;
    int maximize;//bool
    //  double gamma { get; set; }
    double maximizer;

    //BitmapImage previewPic;
    int previewFrame;
    unsigned char errorString[128];

    int isMLV;//bool

    unsigned char DNGHeader[7728];
    unsigned char version[64];
    unsigned char versionString[128];
    unsigned char propertiesString[256];
};

struct audiodata
{
    bool hasAudio;
    std::string audioFile;
    int audioSamplingRate;
    int audioBitsPerSample;
    int audioFormat;
    int audioChannels;
    int audioBytesPerSecond;
    int audioBlockAlign;
};

/*
struct threaddata
{
    int frame;
    CountdownEvent CDEvent;
};*/
struct lensdata
{
    std::string lens;
    int focalLength;
    int aperture;
    int isoValue;
    std::string shutter;
};

#pragma pack(1)
struct mlvBlock
{
    unsigned char blockTag[4];
    long fileOffset;
    int fileNo;
    int blockLength;
    long timestamp;
    int EDMACoffset;
    int MLVFrameNo;
};

#pragma pack(1)
struct rawBlock
{
    int fileNo;
    long fileOffset;
    int splitted;//bool
};


struct Blocks
{
    std::list<mlvBlock> mlvBlockList;
    std::list<rawBlock> rawBlockList;
};

struct filedata
{
    bool convertIt;
/*    std::string fileName;
    std::string fileNameOnly;
    std::string fileNameShort;
    std::string fileNameNum;
    std::string tempPath;
    std::string sourcePath;
    std::string basePath;
    std::string destinationPath;
    std::string extraPathInfo;
    std::string _changedPath;
    std::string outputFilename;
*/
//    DateTime creationTime;
//    DateTime modificationTime;

    mlvBlock VIDFBlock;
    rawBlock RAWBlock;

};

struct convertSettings
{
    int bitdepth;
    bool maximize;
    double maximizeValue;
    std::string format;
    bool pinkHighlight;
    bool chromaSmoothing;
    bool verticalBanding;
    bool proxyJpegs;

};

struct datatype
{
    unsigned char* rawData;
    metadata metaData;
    audiodata audioData;
    filedata fileData;
    lensdata lensData;
  //  threaddata threadData { get; set; }
    convertSettings convertData;
};

struct rawtype
{
    datatype data;
    double* verticalStripes;
    bool verticalBandingNeeded;
    std::list<rawBlock> RAWBlocks;
    std::list<mlvBlock> VIDFBlocks;
    std::list<mlvBlock> AUDFBlocks;
};


//typedef std::unique_ptr<unsigned char[]> unique_data_block;
//typedef ArrayPtr<unsigned char> unique_data_block;
typedef ManagedPtr<unsigned char> managed_data_block;

#endif
