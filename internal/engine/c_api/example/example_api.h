#ifndef EXAMPLE_API_H
#define EXAMPLE_API_H

#ifdef __cplusplus
extern "C" {
#endif

// 导出函数声明
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

EXPORT int add(int a, int b);
EXPORT double multiply(double a, double b);
EXPORT char *get_message(void);

#ifdef __cplusplus
}
#endif

#endif // EXAMPLE_API_H