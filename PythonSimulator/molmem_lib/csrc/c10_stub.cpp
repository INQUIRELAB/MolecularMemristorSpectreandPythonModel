// C10 stub for Windows PyTorch ROCm compilation
#include <string>
#include <cstdint>

#ifndef _MSC_VER
#define __declspec(dllexport)
#endif

namespace c10 {
    struct SourceLocation {
        const char* file;
        uint32_t line;
    };
    
    class __declspec(dllexport) Error {
    public:
        Error(SourceLocation loc, std::string msg) : file_(loc.file), line_(loc.line), msg_(msg) {}
        virtual ~Error() = default;
    private:
        const char* file_;
        uint32_t line_;
        std::string msg_;
    };
    
    class __declspec(dllexport) ValueError : public Error {
    public:
        ValueError(SourceLocation loc, std::string msg) : Error(loc, msg) {}
    };
}
