/* obsConsoleObserverService.h
   Generated by gSOAP 2.8.3 from obsModHeader.h

Copyright(C) 2000-2011, Robert van Engelen, Genivia Inc. All Rights Reserved.
The generated code is released under one of the following licenses:
1) GPL or 2) Genivia's license for commercial use.
This program is released under the GPL with the additional exemption that
compiling, linking, and/or using OpenSSL is allowed.
*/

#ifndef obsConsoleObserverService_H
#define obsConsoleObserverService_H
#include "obsH.h"

namespace obs {
class SOAP_CMAC ConsoleObserverService : public soap
{ public:
	/// Constructor
	ConsoleObserverService();
	/// Constructor with copy of another engine state
	ConsoleObserverService(const struct soap&);
	/// Constructor with engine input+output mode control
	ConsoleObserverService(soap_mode iomode);
	/// Constructor with engine input and output mode control
	ConsoleObserverService(soap_mode imode, soap_mode omode);
	/// Destructor, also frees all deserialized data
	virtual ~ConsoleObserverService();
	/// Delete all deserialized data (uses soap_destroy and soap_end)
	virtual	void destroy();
	/// Initializer used by constructor
	virtual	void ConsoleObserverService_init(soap_mode imode, soap_mode omode);
	/// Create a copy
	virtual	ConsoleObserverService *copy() SOAP_PURE_VIRTUAL;
	/// Force close connection (normally automatic)
	virtual	int soap_close_socket();
	/// Return sender-related fault to sender
	virtual	int soap_senderfault(const char *string, const char *detailXML);
	/// Return sender-related fault with SOAP 1.2 subcode to sender
	virtual	int soap_senderfault(const char *subcodeQName, const char *string, const char *detailXML);
	/// Return receiver-related fault to sender
	virtual	int soap_receiverfault(const char *string, const char *detailXML);
	/// Return receiver-related fault with SOAP 1.2 subcode to sender
	virtual	int soap_receiverfault(const char *subcodeQName, const char *string, const char *detailXML);
	/// Print fault
	virtual	void soap_print_fault(FILE*);
#ifndef WITH_LEAN
	/// Print fault to stream
	virtual	void soap_stream_fault(std::ostream&);
	/// Put fault into buffer
	virtual	char *soap_sprint_fault(char *buf, size_t len);
#endif
	/// Disables and removes SOAP Header from message
	virtual	void soap_noheader();
	/// Get SOAP Header structure (NULL when absent)
	virtual	const SOAP_ENV__Header *soap_header();
	/// Run simple single-thread iterative service on port until a connection error occurs (returns error code or SOAP_OK), use this->bind_flag = SO_REUSEADDR to rebind for a rerun
	virtual	int run(int port);
	/// Bind service to port (returns master socket or SOAP_INVALID_SOCKET)
	virtual	SOAP_SOCKET bind(const char *host, int port, int backlog);
	/// Accept next request (returns socket or SOAP_INVALID_SOCKET)
	virtual	SOAP_SOCKET accept();
	/// Serve this request (returns error code or SOAP_OK)
	virtual	int serve();
	/// Used by serve() to dispatch a request (returns error code or SOAP_OK)
	virtual	int dispatch();

	///
	/// Service operations (you should define these):
	/// Note: compile with -DWITH_PURE_VIRTUAL for pure virtual methods
	///

	/// Web service operation 'ontologySelected' (returns error code or SOAP_OK)
	virtual	int ontologySelected(LONG64 rootId, struct fw__ontologySelectedResponse &_param_1) SOAP_PURE_VIRTUAL;

	/// Web service operation 'ontologyChanged' (returns error code or SOAP_OK)
	virtual	int ontologyChanged(LONG64 rootId, struct fw__ontologyChangedResponse &_param_2) SOAP_PURE_VIRTUAL;

	/// Web service operation 'entitySelected' (returns error code or SOAP_OK)
	virtual	int entitySelected(std::string _category, std::string _entityId, bool _clearAll, struct fw__entitySelectedResponse &_param_3) SOAP_PURE_VIRTUAL;

	/// Web service operation 'entityDeselected' (returns error code or SOAP_OK)
	virtual	int entityDeselected(std::string _category, std::string _entityId, struct fw__entityDeselectedResponse &_param_4) SOAP_PURE_VIRTUAL;

	/// Web service operation 'entityChanged' (returns error code or SOAP_OK)
	virtual	int entityChanged(LONG64 entityId, struct fw__entityChangedResponse &_param_5) SOAP_PURE_VIRTUAL;

	/// Web service operation 'entityViewRequested' (returns error code or SOAP_OK)
	virtual	int entityViewRequested(LONG64 entityId, struct fw__entityViewRequestedResponse &_param_6) SOAP_PURE_VIRTUAL;

	/// Web service operation 'annotationsChanged' (returns error code or SOAP_OK)
	virtual	int annotationsChanged(LONG64 entityId, struct fw__annotationsChangedResponse &_param_7) SOAP_PURE_VIRTUAL;

	/// Web service operation 'sessionSelected' (returns error code or SOAP_OK)
	virtual	int sessionSelected(LONG64 sessionId, struct fw__sessionSelectedResponse &_param_8) SOAP_PURE_VIRTUAL;

	/// Web service operation 'sessionDeselected' (returns error code or SOAP_OK)
	virtual	int sessionDeselected(struct fw__sessionDeselectedResponse &_param_9) SOAP_PURE_VIRTUAL;
};

} // namespace obs

#endif
