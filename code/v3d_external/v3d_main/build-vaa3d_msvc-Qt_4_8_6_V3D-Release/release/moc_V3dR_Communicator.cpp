/****************************************************************************
** Meta object code from reading C++ file 'V3dR_Communicator.h'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.6)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../vrrenderer/V3dR_Communicator.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'V3dR_Communicator.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.6. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_V3dR_Communicator[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
      35,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
      25,       // signalCount

 // signals: signature, parameters, type, tag, flags
      19,   18,   18,   18, 0x05,
      33,   18,   18,   18, 0x05,
      53,   18,   18,   18, 0x05,
      75,   18,   18,   18, 0x05,
      94,   18,   18,   18, 0x05,
     116,   18,   18,   18, 0x05,
     137,  135,   18,   18, 0x05,
     157,   18,   18,   18, 0x05,
     178,  135,   18,   18, 0x05,
     198,   18,   18,   18, 0x05,
     216,  135,   18,   18, 0x05,
     243,  135,   18,   18, 0x05,
     275,   18,   18,   18, 0x05,
     294,   18,   18,   18, 0x05,
     319,  316,   18,   18, 0x05,
     346,   18,   18,   18, 0x05,
     366,   18,   18,   18, 0x05,
     393,   18,   18,   18, 0x05,
     417,   18,   18,   18, 0x05,
     448,   18,   18,   18, 0x05,
     455,   18,   18,   18, 0x05,
     470,   18,   18,   18, 0x05,
     494,   18,   18,   18, 0x05,
     515,   18,   18,   18, 0x05,
     536,   18,   18,   18, 0x05,

 // slots: signature, parameters, type, tag, flags
     556,  552,   18,   18, 0x0a,
     575,  552,   18,   18, 0x0a,
     599,  552,   18,   18, 0x0a,
     626,  552,   18,   18, 0x0a,
     650,   18,   18,   18, 0x0a,
     664,   18,   18,   18, 0x0a,
     678,   18,   18,   18, 0x0a,
     689,   18,   18,   18, 0x0a,
     719,   18,   18,   18, 0x0a,
     743,  739,   18,   18, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_V3dR_Communicator[] = {
    "V3dR_Communicator\0\0load(QString)\0"
    "reloadFile(QString)\0msgtoprocess(QString)\0"
    "msgtowarn(QString)\0msgtoanalyze(QString)\0"
    "msgtosend(QString)\0,\0addSeg(QString,int)\0"
    "addManySegs(QString)\0delSeg(QString,int)\0"
    "splitSeg(QString)\0addMarker(QString,QString)\0"
    "addManyMarkers(QString,QString)\0"
    "delMarker(QString)\0retypeMarker(QString)\0"
    ",,\0retypeSeg(QString,int,int)\0"
    "connectSeg(QString)\0updateOnlineUsers(QString)\0"
    "updateBCIstate(QString)\0"
    "setDefineSomaActionState(bool)\0exit()\0"
    "updateQcInfo()\0updateQcMarkersCounts()\0"
    "updateQcSegsCounts()\0addManyMarkersDone()\0"
    "addMarkerDone()\0msg\0TFProcess(QString)\0"
    "processWarnMsg(QString)\0"
    "processAnalyzeMsg(QString)\0"
    "processSendMsg(QString)\0onReadyRead()\0"
    "onConnected()\0autoExit()\0"
    "resetWarnMulBifurcationFlag()\0"
    "resetWarnLoopFlag()\0MSG\0UpdateBCIMsg(QString)\0"
};

void V3dR_Communicator::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        V3dR_Communicator *_t = static_cast<V3dR_Communicator *>(_o);
        switch (_id) {
        case 0: _t->load((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 1: _t->reloadFile((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 2: _t->msgtoprocess((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 3: _t->msgtowarn((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 4: _t->msgtoanalyze((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 5: _t->msgtosend((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 6: _t->addSeg((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 7: _t->addManySegs((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 8: _t->delSeg((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 9: _t->splitSeg((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 10: _t->addMarker((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2]))); break;
        case 11: _t->addManyMarkers((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2]))); break;
        case 12: _t->delMarker((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 13: _t->retypeMarker((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 14: _t->retypeSeg((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3]))); break;
        case 15: _t->connectSeg((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 16: _t->updateOnlineUsers((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 17: _t->updateBCIstate((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 18: _t->setDefineSomaActionState((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 19: _t->exit(); break;
        case 20: _t->updateQcInfo(); break;
        case 21: _t->updateQcMarkersCounts(); break;
        case 22: _t->updateQcSegsCounts(); break;
        case 23: _t->addManyMarkersDone(); break;
        case 24: _t->addMarkerDone(); break;
        case 25: _t->TFProcess((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 26: _t->processWarnMsg((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 27: _t->processAnalyzeMsg((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 28: _t->processSendMsg((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 29: _t->onReadyRead(); break;
        case 30: _t->onConnected(); break;
        case 31: _t->autoExit(); break;
        case 32: _t->resetWarnMulBifurcationFlag(); break;
        case 33: _t->resetWarnLoopFlag(); break;
        case 34: _t->UpdateBCIMsg((*reinterpret_cast< QString(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData V3dR_Communicator::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject V3dR_Communicator::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_V3dR_Communicator,
      qt_meta_data_V3dR_Communicator, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &V3dR_Communicator::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *V3dR_Communicator::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *V3dR_Communicator::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_V3dR_Communicator))
        return static_cast<void*>(const_cast< V3dR_Communicator*>(this));
    return QObject::qt_metacast(_clname);
}

int V3dR_Communicator::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 35)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 35;
    }
    return _id;
}

// SIGNAL 0
void V3dR_Communicator::load(QString _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void V3dR_Communicator::reloadFile(QString _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void V3dR_Communicator::msgtoprocess(QString _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void V3dR_Communicator::msgtowarn(QString _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}

// SIGNAL 4
void V3dR_Communicator::msgtoanalyze(QString _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 4, _a);
}

// SIGNAL 5
void V3dR_Communicator::msgtosend(QString _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 5, _a);
}

// SIGNAL 6
void V3dR_Communicator::addSeg(QString _t1, int _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 6, _a);
}

// SIGNAL 7
void V3dR_Communicator::addManySegs(QString _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 7, _a);
}

// SIGNAL 8
void V3dR_Communicator::delSeg(QString _t1, int _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 8, _a);
}

// SIGNAL 9
void V3dR_Communicator::splitSeg(QString _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 9, _a);
}

// SIGNAL 10
void V3dR_Communicator::addMarker(QString _t1, QString _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 10, _a);
}

// SIGNAL 11
void V3dR_Communicator::addManyMarkers(QString _t1, QString _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 11, _a);
}

// SIGNAL 12
void V3dR_Communicator::delMarker(QString _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 12, _a);
}

// SIGNAL 13
void V3dR_Communicator::retypeMarker(QString _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 13, _a);
}

// SIGNAL 14
void V3dR_Communicator::retypeSeg(QString _t1, int _t2, int _t3)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)), const_cast<void*>(reinterpret_cast<const void*>(&_t3)) };
    QMetaObject::activate(this, &staticMetaObject, 14, _a);
}

// SIGNAL 15
void V3dR_Communicator::connectSeg(QString _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 15, _a);
}

// SIGNAL 16
void V3dR_Communicator::updateOnlineUsers(QString _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 16, _a);
}

// SIGNAL 17
void V3dR_Communicator::updateBCIstate(QString _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 17, _a);
}

// SIGNAL 18
void V3dR_Communicator::setDefineSomaActionState(bool _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 18, _a);
}

// SIGNAL 19
void V3dR_Communicator::exit()
{
    QMetaObject::activate(this, &staticMetaObject, 19, 0);
}

// SIGNAL 20
void V3dR_Communicator::updateQcInfo()
{
    QMetaObject::activate(this, &staticMetaObject, 20, 0);
}

// SIGNAL 21
void V3dR_Communicator::updateQcMarkersCounts()
{
    QMetaObject::activate(this, &staticMetaObject, 21, 0);
}

// SIGNAL 22
void V3dR_Communicator::updateQcSegsCounts()
{
    QMetaObject::activate(this, &staticMetaObject, 22, 0);
}

// SIGNAL 23
void V3dR_Communicator::addManyMarkersDone()
{
    QMetaObject::activate(this, &staticMetaObject, 23, 0);
}

// SIGNAL 24
void V3dR_Communicator::addMarkerDone()
{
    QMetaObject::activate(this, &staticMetaObject, 24, 0);
}
QT_END_MOC_NAMESPACE
