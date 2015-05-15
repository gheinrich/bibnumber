/** includes */
#include <iostream>

#ifndef LOG_H
#define LOG_H

/** public macros */
#define LOG_CHAINS (1<<0)
#define LOG_TXT_ORIENT (1<<1)
#define LOG_COMPONENTS (1<<2)
#define LOG_TEXTREC (1<<3)
#define LOG_SVM (1<<4)
#define LOG_COMP_PAIRS (1<<5)
#define LOG_SYMM_CHECK (1<<6)
#define LOG_ALL (0xFFFFFFFF)
#define LOG_NONE (0)

#define LOG_MASK (biblog::log_mask)

#define LOG(mask,x) do { \
  if (LOG_MASK & (mask)) { std::cout << x ; } \
} while (0)

#define LOGL(mask,x) do { \
  if (LOG_MASK & (mask)) { std::cout << x << std::endl; } \
} while (0)

namespace biblog
{
	/** public variables */
	extern int log_mask;

	/** public functions */
	void set_log_mask(int log_mask);
}

#endif /* #ifndef LOG_H */
