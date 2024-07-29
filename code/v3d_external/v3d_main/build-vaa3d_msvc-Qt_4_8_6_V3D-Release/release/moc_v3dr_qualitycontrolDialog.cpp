/****************************************************************************
** Meta object code from reading C++ file 'v3dr_qualitycontrolDialog.h'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.6)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../3drenderer/v3dr_qualitycontrolDialog.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'v3dr_qualitycontrolDialog.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.6. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_V3dr_qualitycontrolDialog[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       4,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      35,   27,   26,   26, 0x0a,
      63,   26,   26,   26, 0x0a,
      85,   26,   26,   26, 0x0a,
     104,   26,   26,   26, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_V3dr_qualitycontrolDialog[] = {
    "V3dr_qualitycontrolDialog\0\0row,col\0"
    "doubleClickHandler(int,int)\0"
    "updateMarkersCounts()\0updateSegsCounts()\0"
    "updateInfo()\0"
};

void V3dr_qualitycontrolDialog::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        V3dr_qualitycontrolDialog *_t = static_cast<V3dr_qualitycontrolDialog *>(_o);
        switch (_id) {
        case 0: _t->doubleClickHandler((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 1: _t->updateMarkersCounts(); break;
        case 2: _t->updateSegsCounts(); break;
        case 3: _t->updateInfo(); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData V3dr_qualitycontrolDialog::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject V3dr_qualitycontrolDialog::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_V3dr_qualitycontrolDialog,
      qt_meta_data_V3dr_qualitycontrolDialog, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &V3dr_qualitycontrolDialog::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *V3dr_qualitycontrolDialog::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *V3dr_qualitycontrolDialog::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_V3dr_qualitycontrolDialog))
        return static_cast<void*>(const_cast< V3dr_qualitycontrolDialog*>(this));
    return QWidget::qt_metacast(_clname);
}

int V3dr_qualitycontrolDialog::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 4)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 4;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
