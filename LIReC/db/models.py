# coding: utf-8
from enum import Enum
from sqlalchemy import ARRAY, CHAR, Column, DateTime, Float, ForeignKey, Integer, Numeric, String, UniqueConstraint, text, Table
from sqlalchemy.sql.sqltypes import BigInteger
from sqlalchemy.types import Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship, deferred
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata


class Priority(Enum): # TODO fill with values...
    PROCEDURAL = 1 # procedurally-generated, low priority
    MANUAL = 5 # manually inserted, high priority


class Constant(Base):
    __tablename__ = 'constant'

    const_id = Column(UUID, primary_key=True, server_default=text('uuid_generate_v1()'))
    value = Column(Numeric)
    precision = Column(Integer)
    time_added = Column(DateTime, server_default=text('CURRENT_TIMESTAMP'))
    source = Column(ForeignKey('source.source_id'))
    source_notes = Column(String)
    priority = Column(Integer, nullable=False, server_default=text('1'))
    tweeted = Column(Integer, nullable=False, server_default=text('0'))
    
    source_ref = relationship('Source', lazy='subquery')


class NamedConstant(Base):
    __tablename__ = 'named_constant'

    const_id = Column(ForeignKey('constant.const_id'), primary_key=True)
    name = Column(String, nullable=False, unique=True)
    description = Column(String)
    
    base = relationship('Constant', lazy='subquery')


class PcfConvergence(Enum):
    ZERO_DENOM = 0 # now considered an illegal PCF
    NO_FR = 1 # now considered an illegal PCF
    INDETERMINATE_FR = 2
    FR = 3
    RATIONAL = 4


class PcfCanonicalConstant(Base):
    __tablename__ = 'pcf_canonical_constant'
    __table_args__ = (
        UniqueConstraint('P', 'Q'),
    )

    const_id = Column(ForeignKey('constant.const_id'), primary_key=True)
    #original_a = Column(ARRAY(Numeric()))
    #original_b = Column(ARRAY(Numeric()))
    P = Column(ARRAY(Numeric()), nullable=False)
    Q = Column(ARRAY(Numeric()), nullable=False)
    last_matrix = deferred(Column(Text())) # don't always need to load this!
    depth = Column(Integer)
    convergence = Column(Integer)
    
    base = relationship('Constant', lazy='subquery')


class DerivedConstant(Base):
    __tablename__ = 'derived_constant'

    const_id = Column(ForeignKey('constant.const_id'), primary_key=True)
    family = Column(String, nullable=False)
    args = Column(JSONB(astext_type=Text()), nullable=False)
    
    base = relationship('Constant', lazy='subquery')


class PowerOfConstant(Base):
    __tablename__ = 'power_of_constant'

    const_id = Column(ForeignKey('constant.const_id'), primary_key=True)
    based_on = Column(ForeignKey('constant.const_id'))
    power = Column(Integer, nullable=False)
    
    base = relationship('Constant', lazy='subquery', foreign_keys=[const_id])


class PcfFamily(Base):
    __tablename__ = 'pcf_family'
    __table_args__ = (
        UniqueConstraint('a', 'b'),
    )

    family_id = Column(UUID, primary_key=True, server_default=text('uuid_generate_v1()'))
    a = Column(String, nullable=False)
    b = Column(String, nullable=False)


class ScanHistory(Base):
    __tablename__ = 'scan_history'

    const_id = Column(ForeignKey('constant.const_id'), primary_key=True)
    algorithm = Column(String, nullable=False)
    time_scanned = Column(DateTime, server_default=text('CURRENT_TIMESTAMP'))
    details = Column(String)
    
    base = relationship('Constant')


constant_in_relation_table = Table(
    'constant_in_relation',
    Base.metadata,
    Column('const_id', ForeignKey('constant.const_id', onupdate='CASCADE', ondelete='CASCADE'), primary_key=True),
    Column('relation_id', ForeignKey('relation.relation_id', onupdate='CASCADE', ondelete='CASCADE'), primary_key=True),
)


class Relation(Base):
    __tablename__ = 'relation'

    relation_id = Column(UUID, primary_key=True, server_default=text('uuid_generate_v1()'))
    relation_type = Column(String, nullable=False)
    # if this needs an order on the constants and cfs (and it probably will),
    # it is determined by ascending order on the const_ids
    details = Column(ARRAY(Integer()), nullable=False)
    precision = Column(Integer)
    time_added = Column(DateTime, server_default=text('CURRENT_TIMESTAMP'))
    priority = Column(Integer, nullable=False, server_default=text('1'))
    tweeted = Column(Integer, nullable=False, server_default=text('0'))

    constants = relationship('Constant', secondary=constant_in_relation_table)


class Source(Base):
    __tablename__ = 'source'

    source_id = Column(UUID, primary_key=True, server_default=text('uuid_generate_v1()'))
    alias = Column(String, unique=True)
    reference = Column(String, unique=True)
    link = Column(String, unique=True)
