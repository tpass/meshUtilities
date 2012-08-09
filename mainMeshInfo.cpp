//@HEADER
/*
*******************************************************************************

    Copyright (C) 2004, 2005, 2007 EPFL, Politecnico di Milano, INRIA
    Copyright (C) 2010 EPFL, Politecnico di Milano, Emory University

    This file is part of LifeV.

    LifeV is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    LifeV is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with LifeV.  If not, see <http://www.gnu.org/licenses/>.

*******************************************************************************
*/
//@HEADER

/*!
    @file
    @brief

    @author Tiziano Passerini <tiziano@mathcs.emory.edu>
    @date 03-02-2011
 */

// Tell the compiler to ignore specific kind of warnings:
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include <Epetra_ConfigDefs.h>
#ifdef EPETRA_MPI
#include <mpi.h>
#include <Epetra_MpiComm.h>
#else
#include <Epetra_SerialComm.h>
#endif

//Tell the compiler to restore the warning previously silented
#pragma GCC diagnostic warning "-Wunused-variable"
#pragma GCC diagnostic warning "-Wunused-parameter"

#include <lifev/core/LifeV.hpp>
#include <lifev/core/util/Displayer.hpp>

#include <lifev/core/filter/ExporterEnsight.hpp>
//#include <lifev/core/filter/ExporterHDF5.hpp>

#include <lifev/core/fem/FESpace.hpp>
#include <lifev/core/fem/PostProcessingBoundary.hpp>

#include <lifev/core/mesh/MeshPartitioner.hpp>
#include <lifev/core/mesh/RegionMesh3DStructured.hpp>
#include <lifev/core/mesh/RegionMesh.hpp>

#include <lifev/core/mesh/MeshData.hpp>

#include <lifev/ecm2/mesh/MeshUtils.hpp>

using namespace LifeV;


typedef RegionMesh<LinearTetra>       mesh_Type;
typedef VectorEpetra                    vector_Type;
typedef FESpace< mesh_Type, MapEpetra > feSpace_Type;
typedef boost::shared_ptr<feSpace_Type> feSpacePtr_Type;
typedef boost::shared_ptr<Epetra_Comm>  commPtr_Type;
typedef ExporterEnsight<mesh_Type>      exporter_Type;


int
main( int argc, char** argv )
{

#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
    commPtr_Type commPtr(new Epetra_MpiComm(MPI_COMM_WORLD));
#else
    commPtr_Type commPtr(new Epetra_SerialComm);
#endif

    Displayer displayer(commPtr);

    // ****************************
    // Read first the data needed
    // ****************************
    displayer.leaderPrint(" -- Reading the data ... " );
    GetPot dataFile( "data" );
    displayer.leaderPrint( " done ! \n" );

    std::list<UInt> flagList;
    parseList(dataFile( "meshInfo/flagList", ""), flagList);

    // ****************************
    // Build and partition the mesh
    // ****************************
    MeshData meshData( dataFile, "mesh" );
    std::string displayerString = "Reading mesh from file " + meshData.meshFile();
    displayer.leaderPrint( std::setiosflags(std::ios::left), std::setw(70), displayerString );
    displayer.leaderPrint( std::resetiosflags(std::ios::right), std::setw(5), "[OK]\n" ); //,

    boost::shared_ptr< mesh_Type > fullMeshPtr(new RegionMesh<LinearTetra>);

    readMesh(*fullMeshPtr,meshData);

    displayer.leaderPrint( " done ! \n" );

    // wr_medit_ascii( "globalMesh.mesh", *fullMeshPtr );

    displayer.leaderPrint( " -- Marker of the global mesh ... ", fullMeshPtr->marker(), "\n" );

    // fullMeshPtr->globalToLocalFace().size();

    // ****************************
    // Build the FESpace
    // ****************************
    displayer.leaderPrint( " -- Building FESpace ... " );
    std::string feSpace(dataFile("discretization/feSpace","P1"));

    feSpacePtr_Type feSpacePtr( new feSpace_Type( fullMeshPtr, feSpace, nDimensions, commPtr) );

    displayer.leaderPrint( " done ! \n" );
    displayer.leaderPrint( " ---> Dofs: ", feSpacePtr->dof().numTotalDof(), "\n" );

    // ****************************
    // Compute interesting quantities
    // ****************************
    Vector meshSizesVector = meshSizes( *fullMeshPtr, feSpacePtr->fe(), commPtr );

    displayer.leaderPrint( "\tmax mesh size: ", meshSizesVector[0], "\n" );
    displayer.leaderPrint( "\tavg mesh size: ", meshSizesVector[1], "\n" );
    displayer.leaderPrint( "\tmin mesh size: ", meshSizesVector[2], "\n" );

    for(std::list<UInt>::iterator it=flagList.begin();it!=flagList.end();++it)
    {
        displayer.leaderPrint( " -- Info for label ", *it, "\n" );

        std::vector<ID> faceList;
        faceList.clear();
        // fullMeshPtr->extractEntityList(faceList, FACE, *it);
        Predicates::EntityMarkerIDInterrogator<mesh_Type::face_Type> tmpPredicate( *it );
        faceList = fullMeshPtr->faceList.extractIdAccordingToPredicate(tmpPredicate);

        commPtr->Barrier();
        std::cout << "\tProcess " << commPtr->MyPID() << " sees " << faceList.size() << " faces " << std::endl;
        commPtr->Barrier();

        Vector normalVector = compute_normal( *fullMeshPtr, *it, commPtr );
        Vector centerOfMass = compute_center_of_mass( *fullMeshPtr, FACE, *it, commPtr );
        Real faceArea       = compute_area( *fullMeshPtr, *it, commPtr );

        displayer.leaderPrint( "\tnormalVector:" );
        for( UInt iComp = 0; iComp < nDimensions; ++iComp ) displayer.leaderPrint( " ", normalVector[iComp] );
        displayer.leaderPrint( "\n" );

        displayer.leaderPrint( "\tcenterOfMass:" );
        for( UInt iComp = 0; iComp < nDimensions; ++iComp ) displayer.leaderPrint( " ", centerOfMass[iComp] );
        displayer.leaderPrint( "\n" );

        displayer.leaderPrint( "\tfaceArea: ", faceArea, "\n" );

}


#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return( EXIT_SUCCESS );
}


