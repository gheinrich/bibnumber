/** includes */
#include "debug.h"

/* macros */
//#define DEFAULT_DBG_MASK   ( DBG_TXT_ORIENT | DBG_COMPONENTS )
#define DEFAULT_DBG_MASK   ( DBG_TXT_ORIENT | DBG_CHAINS )
//#define DEFAULT_DBG_MASK   ( DBG_TXT_ORIENT )
//#define DEFAULT_DBG_MASK   ( DBG_ALL )
//#define DEFAULT_DBG_MASK   ( DBG_NONE )


namespace debug
{
	/** public variables */
	int debug_mask = DEFAULT_DBG_MASK;

	/** public functions */
	void set_debug_mask(int mask)
	{
		debug_mask = mask;
	}
}


