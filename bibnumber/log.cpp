/** includes */
#include "log.h"

/* macros */
//#define DEFAULT_DBG_MASK   ( LOG_TEXTREC | LOG_COMPONENTS | LOG_TXT_ORIENT | LOG_SVM)
//#define DEFAULT_DBG_MASK   ( LOG_TEXTREC | LOG_COMPONENTS | LOG_CHAINS | LOG_TXT_ORIENT | LOG_SVM)
//#define DEFAULT_DBG_MASK   ( LOG_TEXTREC | LOG_COMPONENTS | LOG_COMP_PAIRS | LOG_CHAINS | LOG_TXT_ORIENT | LOG_SVM)
#define DEFAULT_DBG_MASK   ( LOG_TEXTREC | LOG_COMPONENTS |  LOG_CHAINS | LOG_TXT_ORIENT | LOG_SVM)
//#define DEFAULT_DBG_MASK   ( DBG_TEXTREC | DBG_COMPONENTS | DBG_CHAINS )
//#define DEFAULT_DBG_MASK   ( DBG_TXT_ORIENT | DBG_CHAINS )
//#define DEFAULT_DBG_MASK   ( DBG_TXT_ORIENT )
//#define DEFAULT_DBG_MASK   ( LOG_ALL )
//#define DEFAULT_DBG_MASK   ( DBG_NONE )


namespace biblog
{
	/** public variables */
	int log_mask = DEFAULT_DBG_MASK;

	/** public functions */
	void set_log_mask(int mask)
	{
		log_mask = mask;
	}
}


