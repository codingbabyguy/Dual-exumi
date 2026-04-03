#ifndef RM_API_GLOBAL_H
#define RM_API_GLOBAL_H

#ifdef __linux
#define RM_APISHARED_EXPORT
#endif
#if _WIN32
#if defined(RM_API_LIBRARY)
#define RM_APISHARED_EXPORT __declspec(dllexport)
#else
#define RM_APISHARED_EXPORT __declspec(dllimport)
#endif
#endif

#endif // RM_API_GLOBAL_H
