/** includes */
#include <iostream>

/** public variables */
extern int debug_mask;

/** public macros */
#define DBG_CHAINS (1<<0)
#define DBG_TXT_ORIENT (1<<1)
#define DBG_COMPONENTS (1<<2)
#define DBG_TEXTREC (1<<3)
#define DBG_ALL (0xFFFFFFFF)
#define DBG_NONE (0)

#define DBG_MASK (debug::debug_mask)

#define DBG(mask,x) do { \
  if (DBG_MASK & (mask)) { std::cout << x ; } \
} while (0)

#define DBGL(mask,x) do { \
  if (DBG_MASK & (mask)) { std::cout << x << std::endl; } \
} while (0)

namespace debug
{
	/** public variables */
	extern int debug_mask;

	/** public functions */
	void set_debug_mask(int debug_mask);
}
