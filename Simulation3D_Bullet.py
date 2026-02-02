import os
import sys
import time
import abaqusConstants
import mesh
import numpy           as np
import json
import math
import pandas          as pd
import pathlib
import shutil
from time import sleep
    
    
from abaqus      import *
from driverUtils import *
from caeModules  import *
    
    
def log(message):
    print(message, file = sys.__stdout__)
    return
    
    
class Simulation3D():
    
    def __init__( self,
                  PLATE_WIDTH  = 40,
                  PLATE_HEIGHT = 2.5 ):
        
        #***********
        # PARAMETERS
        #***********
        self.index                 = None
        self.bullet_speed          = None
        self.bullet_speed_x        = None
        self.bullet_speed_y        = None
        self.bullet_speed_z        = None
        self.bullet_radius         = None
        self.bullet_impact_angle_x = None
        self.bullet_impact_angle_y = None
        self.bullet_impact_angle_z = None
        
        
        #*****************
        # OBJECT DIMENSION
        #*****************
        self.plate_width        = PLATE_WIDTH
        self.plate_height       = PLATE_HEIGHT
        self.bullet_base_radius = None   # la seed size la calcolo usando questo come riferimento
        
        
        #******************
        # INITIAL POSITIONS
        #******************
        self.plate_origin_x  = 0
        self.plate_origin_y  = 0
        self.plate_origin_z  = 0
        self.bullet_origin_x = 0
        self.bullet_origin_y = 0
        self.bullet_origin_z = 0
        
        
        #***************
        # MATERIAL: LEAD
        #***************
        self.lead_density = ( (1.153e-5,   ), )
        self.lead_elastic = ( (1.4e4, 0.42 ), )
        
        
        #******************************
        # MATERIAL: 304 STAINLESS STEEL
        #******************************
        self.steel_density = ( (8e-6,         ), )  # kg/mm³
        self.steel_elastic = ( (196.5e3,   0.3), )  # (YOUNG'S MODULUS, POISSON RATIO)
        self.steel_plastic = ( (215,         0),
                               (496.8,  0.0975),
                               (687.6,  0.1965),
                               (893.6,  0.2955),
                               (1086.6, 0.3945)  )
        
        
        #***************************
        # MESH PARAMETERS (MODIFIED)
        #***************************
        # self.bullet_seed_size          = 3
        # self.plate_seed_sides_min      = 3
        # self.plate_seed_sides_max      = 5
        # self.plate_seed_top_bottom_min = 3
        # self.plate_seed_top_bottom_max = 5
        
        
        #***************************
        # MESH PARAMETERS (ORIGINAL)
        #***************************
        self.bullet_seed_size          = 1
        self.plate_seed_sides_min      = 0.5
        self.plate_seed_sides_max      = 1
        self.plate_seed_top_bottom_min = 0.5
        self.plate_seed_top_bottom_max = 2
        
        
        #************
        # MISCELLANEA
        #************
        self.time_period      = None     # SIMULATION ELAPSED TIME
        self.output_frequency = 40       # FREQUENCY OUTPUT
        
        
    def _saveInputDataToFile(self):
        
        inputData = { "index"                 : self.index,
                      "bullet_speed_x"        : self.bullet_speed_x,
                      "bullet_speed_y"        : self.bullet_speed_y,
                      "bullet_speed_z"        : self.bullet_speed_z,
                      "bullet_speed"          : self.bullet_speed,
                      "bullet_impact_angle_x" : self.bullet_impact_angle_x,
                      "bullet_impact_angle_y" : self.bullet_impact_angle_y,
                      "bullet_impact_angle_z" : self.bullet_impact_angle_z,
                      "bullet_radius"         : self.bullet_radius }
                          
                          
        # salvo in un file di nome "<index>_input.json"
        filename = os.path.join(self.new_path, self.folder_name + "_input.json")
            
        with open(filename, "w") as outfile:
        
            json.dump(inputData, outfile)
        
        
    def runSimulation( self,
                       BULLET_RADIUS,
                       BULLET_SPEED,
                       BULLET_X_CENTER,
                       BULLET_Y_CENTER,
                       BULLET_Z_CENTER,
                       SIMULATION_ID,
                       LENGTH_SIDE_RATIO,
                       SAVEINPUTDATA        = True, 
                       SAVEBULLETSPEED      = True,
                       SAVEDISPLACEMENT     = True, 
                       SAVECOORDINATES      = True,
                       SAVEDATABASE         = True, 
                       SAVEPLATECOORDINATES = False,
                       SAVEJOBINPUT         = False):
        
        
        #*************************
        # RESETTING THE ENVIRONMENT
        #*************************   
        Mdb()
        
        
        #******************
        # SAVING PARAMETERS
        #******************
        self.index             = SIMULATION_ID
        self.bullet_radius     = BULLET_RADIUS
        self.bullet_speed      = BULLET_SPEED
        self.length_side_ratio = LENGTH_SIDE_RATIO
        
        
        #**************************************
        # INITIAL POSITION OF THE BULLET CENTER
        #**************************************
        self.bullet_origin_x = BULLET_X_CENTER
        self.bullet_origin_y = BULLET_Y_CENTER
        self.bullet_origin_z = BULLET_Z_CENTER
        self.bullet_length   = self.length_side_ratio * self.bullet_radius   # YOU CAN INCREASE/DECREASE (4–8 RECOMMENDED)
        
        
        print( f"Simulation Id   : {self.index:d}"              )               #---> TO DELETE
        print( f"Bullet Origin X : {self.bullet_origin_x:8.4f}" )               #---> TO DELETE
        print( f"Bullet Origin Y : {self.bullet_origin_y:8.4f}" )               #---> TO DELETE
        print( f"Bullet Origin Z : {self.bullet_origin_z:8.4f}" )               #---> TO DELETE
        
        
        # vogliamo che il tempo che ci mette il proiettile a raggiungere la lastra sia sempre 2/30 secondi
        self.TIME_TO_IMPACT = 2 / 30
        
        
        # Crea cartella (se non esiste) con nome <index>
        self.folder_name   = f'Dynamic_Simulation_{self.index}'
        self.previous_path = os.getcwd()
        self.new_path      = os.path.join( self.previous_path, self.folder_name )
        
        
        os.makedirs( name     = self.new_path, 
                     exist_ok = True )
        
        
        #************************
        # CHECKING FOR INPUT DATA
        #************************
        if (self.bullet_origin_y - self.bullet_radius) <= 0:
            
            log('Bullet is too close to the plate.')
            return 0, False
        
        
        #***************************
        # CHANGING WORKING DIRECTORY
        #***************************
        os.chdir(self.new_path)
        
        
        #*****************
        # CREATING A MODEL
        #*****************
        MODEL_NAME = f'Simulation_{self.index}'
        model      = mdb.Model( name = MODEL_NAME )
        
        
        #************************
        # DELETING STANDARD MODEL
        #************************
        del mdb.models['Model-1']
        
        
        #*******************************
        # CREATING A PLATE PART AND SETS
        #*******************************
        sketch_plate = model.ConstrainedSketch( name      = 'sketch-plate', 
                                                sheetSize = self.plate_width )
                                                 
                                                 
        sketch_plate.rectangle( ( -self.plate_width/2,                  0 ), 
                                (  self.plate_width/2, -self.plate_height ) )
        
        
        # crea plate usando lo sketch
        part_plate = model.Part( name           = 'plate', 
                                 dimensionality = abaqusConstants.THREE_D, 
                                 type           = abaqusConstants.DEFORMABLE_BODY )
                                                     
        part_plate.BaseSolidExtrude( sketch = sketch_plate, 
                                     depth  = self.plate_width )
        
        
        # set con tutta la lastra, per il materiale
        part_plate.Set( name  = 'set-all', 
                        cells = part_plate.cells )
        
        # crea diversi set per le superfici della plate, per assegnargli una boundary condition (sotto) e per impostare l'interaction (sopra) 
        # e per gestire il seed della mesh
        part_plate.Set( name = 'surface-top',    faces = part_plate.faces.findAt( coordinates = ( (                  0,                    0,  self.plate_width/2), ) ) )
        part_plate.Set( name = 'surface-bottom', faces = part_plate.faces.findAt( coordinates = ( (                  0, -self.plate_height/1,  self.plate_width/2), ) ) )
        part_plate.Set( name = 'surface-west',   faces = part_plate.faces.findAt( coordinates = ( (-self.plate_width/2, -self.plate_height/2,  self.plate_width/2), ) ) )
        part_plate.Set( name = 'surface-east',   faces = part_plate.faces.findAt( coordinates = ( ( self.plate_width/2, -self.plate_height/2,  self.plate_width/2), ) ) )
        part_plate.Set( name = 'surface-north',  faces = part_plate.faces.findAt( coordinates = ( (                  0, -self.plate_height/2,                   0), ) ) )
        part_plate.Set( name = 'surface-south',  faces = part_plate.faces.findAt( coordinates = ( (                  0, -self.plate_height/2,    self.plate_width), ) ) )
        
        
        # superficie per l'interaction
        part_plate.Surface( name       = 'surface-top', 
                            side1Faces = part_plate.faces.findAt( coordinates = ( (0, 0, self.plate_width/2), )  ) )
        
        
        # set tutte  le superfici esterni, per l'output
        part_plate.Set( name  = 'surface-all', 
                        faces = part_plate.faces )
        
        
        #********************************
        # CREATING A BULLET PART AND SETS
        #********************************
        sketch_bullet = model.ConstrainedSketch( name      = 'sketch-bullet',
                                                 sheetSize = 2 * self.bullet_radius )
        
        sketch_bullet.rectangle( (-self.bullet_radius, -self.bullet_radius),
                                 (+self.bullet_radius, +self.bullet_radius) )
        
        
        # crea il solido estruso (parallelepipedo)
        part_bullet = model.Part( name           = 'bullet',
                                  dimensionality = abaqusConstants.THREE_D,
                                  type           = abaqusConstants.DEFORMABLE_BODY )
        
        
        part_bullet.BaseSolidExtrude( sketch = sketch_bullet,
                                      depth  = self.bullet_length )
        
        
        # set con tutto il proiettile (materiale)
        part_bullet.Set( name  = 'set-all',
                         cells = part_bullet.cells )
        
        
        # superficie per interaction
        part_bullet.Surface( name       = 'surface-all',
                             side1Faces = part_bullet.faces )
        
        
        # set superfici esterne (output)
        part_bullet.Set( name  = 'surface-all',
                         faces = part_bullet.faces )
        
        
        #*******************
        # DEFINING MATERIALS
        #*******************
        
        # PLATE MATERIAL (STEEL)
        material_plate = model.Material( name = 'material-plate' )
        material_plate.Density( table = self.steel_density )
        material_plate.Elastic( table = self.steel_elastic )
        material_plate.Plastic( table = self.steel_plastic )
        
        
        # BULLET MATERIAL (LEAD)
        material_bullet = model.Material(name='material-bullet')
        material_bullet.Density( table = self.lead_density )
        material_bullet.Elastic( table = self.lead_elastic )
        
        
        #******************************************
        # CREATING SECTIONS AND ASSIGNING MATERIALS
        #******************************************
        
        # Sezione per la lastra
        model.HomogeneousSolidSection( name      = 'section-plate',
                                       material  = 'material-plate',
                                       thickness = None )
        
        part_plate.SectionAssignment( region      = part_plate.sets['set-all'],
                                      sectionName = 'section-plate' )
        
        
        # Sezione per il proiettile
        model.HomogeneousSolidSection( name      = 'section-bullet',
                                       material  = 'material-bullet',
                                       thickness = None    )
        
         # Assegna materiale al proiettile
        part_bullet.SectionAssignment( region      = part_bullet.sets['set-all'],
                                       sectionName = 'section-bullet' )
        
        
        #******************
        # CREATING ASSEMBLY
        #******************
        model.rootAssembly.DatumCsysByDefault(abaqusConstants.CARTESIAN)
        
        
        #************************
        # INSTANTIATING THE PLATE
        #************************
        model.rootAssembly.Instance( name      = 'plate',
                                     part      = part_plate,
                                     dependent = abaqusConstants.ON ).translate( vector = (0, 0, -self.plate_width / 2) )
        
        
        #*************************
        # INSTANTIATING THE BULLET
        #*************************
        instance_bullet = model.rootAssembly.Instance( name      = 'bullet',
                                                       part      = part_bullet,
                                                       dependent = abaqusConstants.ON )
        
        #********************
        # ROTATING THE BULLET
        #********************
        instance_bullet.rotateAboutAxis( axisPoint     = (0.0, 0.0, 0.0), 
                                         axisDirection = (1, 0, 0), 
                                         angle         = -90 )
        
        
        #***********************
        # TRANSLATING THE BULLET
        #***********************
        delta_x = self.bullet_origin_x
        delta_y = self.bullet_origin_y - self.bullet_length / 2
        delta_z = self.bullet_origin_z
        instance_bullet.translate( vector = (delta_x, delta_y, delta_z) )
        
        
        # Istanze
        bullet = model.rootAssembly.instances['bullet']
        plate  = model.rootAssembly.instances['plate']
        
        
        # Centri
        # A = np.array( plate.getTranslation() )      # centro della piastra
        # B = np.array( bullet.getTranslation() )     # centro del rettangolo/proiettile
        
        A = np.array( object = [self.plate_origin_x,   self.plate_origin_y,  self.plate_origin_z], dtype = float )
        B = np.array( object = [self.bullet_origin_x, self.bullet_origin_y, self.bullet_origin_z], dtype = float )
        
        
        # Vettore direzione B->A (da proiettile a piastra)
        v    = A - B
        norm = np.linalg.norm(v)
        if norm == 0:
            
            print("Il centro del proiettile e della piastra coincidono")
            
        v /= norm  # vettore target normalizzato
        
        # Vettore iniziale: normale uscente dalla parte corta = +Y
        v0 = np.array([0, 1, 0])
        
        # Calcolo asse di rotazione (perpendicolare a entrambi i vettori)
        rotation_axis = np.cross(v0, v)
        axis_norm     = np.linalg.norm(rotation_axis)
        
        # Caso particolare: v0 e v sono paralleli o antiparalleli
        if axis_norm < 1e-10:
            
            # Calcolo prodotto scalare per determinare se sono paralleli o antiparalleli
            dot = np.dot(v0, v)
            if dot > 0:
                
                # Vettori già paralleli, nessuna rotazione necessaria
                print("Nessuna rotazione necessaria: normale già allineata")
                
            else:
                
                # Vettori antiparalleli: rotazione di 180° attorno a un asse perpendicolare
                # Scegliamo un asse perpendicolare a v0 (es. asse X o Z)
                if abs(v0[0]) < 0.9:
                    
                    rotation_axis = np.cross(v0, np.array([1, 0, 0]))
                    
                else:
                    
                    rotation_axis = np.cross(v0, np.array([0, 0, 1]))
                
                rotation_axis /= np.linalg.norm(rotation_axis)
                theta          = 180.0
                
                bullet.rotateAboutAxis( axisPoint     = B.tolist(),
                                        axisDirection = rotation_axis.tolist(),
                                        angle         = theta )
        else:
            
            # Normalizza l'asse di rotazione
            rotation_axis /= axis_norm
            
            # Calcolo angolo di rotazione
            dot       = np.clip(np.dot(v0, v), -1.0, 1.0)
            theta_rad = math.acos(dot)
            theta_deg = math.degrees(theta_rad)
            
            # Applica la rotazione attorno all'asse calcolato
            bullet.rotateAboutAxis( axisPoint     = B.tolist(),
                                    axisDirection = rotation_axis.tolist(),
                                    angle         = theta_deg )
        
        
        #******************************************
        # CALCULATING SPEED MAGNITUDE OF THE BULLET
        #******************************************
        A                          = np.array( object = [self.plate_origin_x,   self.plate_origin_y,  self.plate_origin_z], dtype = float )
        B                          = np.array( object = [self.bullet_origin_x, self.bullet_origin_y, self.bullet_origin_z], dtype = float )
        direction                  = A - B
        distance                   = np.linalg.norm(direction)
        unit_direction             = direction / distance
        velocity                   = self.bullet_speed * unit_direction
        cosines                    = velocity / self.bullet_speed
        angles                     = np.arccos(cosines)
        self.bullet_impact_angle_x = angles[0]
        self.bullet_impact_angle_y = angles[1]
        self.bullet_impact_angle_z = angles[2]
        self.bullet_speed_x        = velocity[0]
        self.bullet_speed_y        = velocity[1]
        self.bullet_speed_z        = velocity[2]
        
        
        #************************
        # SIMULATION ELAPSED TIME
        #************************
        self.time_period  = self.TIME_TO_IMPACT
        # self.time_period += abs(self.bullet_speed_y / 15000)
        self.time_period += abs( distance/self.bullet_speed)
        
        
        print( f"TIME PERIOD : {self.time_period}" )        #-->TO DELETE
        
        
        #**************************************************
        # DEFINING A STEP
        #**************************************************
        step_1 = model.ExplicitDynamicsStep( name                     = 'Step-1',
                                             previous                 = 'Initial',
                                             description              = 'Dynamic impact of a bullet on plate',
                                             timePeriod               = self.time_period,
                                             timeIncrementationMethod = abaqusConstants.AUTOMATIC_GLOBAL )
        
        
        #**************************************************
        # FIELD OUTPUT
        #**************************************************
        field = model.FieldOutputRequest( name           = 'F-Output-1',
                                          createStepName = 'Step-1',
                                          variables      = ('S', 'E', 'U', 'COORD'),
                                          frequency      = self.output_frequency )
        
        
        #**************************************************
        # CREATING BULLET RIGID BODY CONSTRAINT 
        #**************************************************
        # crea reference point nel centro del proiettile
        RP_bullet_id     = model.rootAssembly.ReferencePoint( point = (delta_x, delta_y, delta_z) ).id
        RP_bullet_region = regionToolset.Region( referencePoints = (model.rootAssembly.referencePoints[RP_bullet_id], ) )
        RP_bullet_set    = model.rootAssembly.Set( name            = "projectile-rp",
                                                   referencePoints = (model.rootAssembly.referencePoints[RP_bullet_id],) )
        
        # assegna rigid body constraint al proiettile
        model.RigidBody( name           = 'constraint-projectile-rigid-body',
                         refPointRegion = RP_bullet_region,
                         bodyRegion     = model.rootAssembly.instances['bullet'].sets['set-all'] )
        
        
        #*******************************************************************
        # CREATING A FILTER TO STOP THE RUN WHEN THE BULLET BOUNCES OR STOPS
        #*******************************************************************
        # crea filtro (Butterworth)
        filter = model.ButterworthFilter( name            = "Filter-1",
                                          cutoffFrequency = 10000,                  # alta, da adattare se necessario
                                          operation       = abaqusConstants.MAX,
                                          limit           = 0.0,                    # ferma se velocità verticale = 0
                                          halt            = True )                  # interrompe l'analisi
        
        # history output per la velocità verticale del centro del proiettile
        model.HistoryOutputRequest( name           = 'H-Output-Bullet-V2',
                                    createStepName = 'Step-1',
                                    region         = RP_bullet_set,
                                    variables      = ('V2',),          # velocità verticale
                                    frequency      = 200,
                                    filter         = "Filter-1" )
        
        
        # ***************************************************
        # DEFINING BOUNDARY CONDITIONS
        # ***************************************************
        bc_left = model.DisplacementBC( name           = 'FixedBC_Left',
                                        createStepName = 'Initial',
                                        region         = model.rootAssembly.instances['plate'].sets['surface-west'],
                                        u1             = 0.0,
                                        u2             = 0.0 )
        
        bc_right = model.DisplacementBC( name           = 'FixedBC_Right',
                                         createStepName = 'Initial',
                                         region         = model.rootAssembly.instances['plate'].sets['surface-east'],
                                         u1             = 0.0,
                                         u2             = 0.0 )
        
        
        #***************************************************
        # PREDEFINED FIELD: INITIAL VELOCITY
        #***************************************************
        velocity = model.Velocity( name      = "Bullet_Velocity",
                                   region    = RP_bullet_region,
                                   velocity1 = self.bullet_speed_x,
                                   velocity2 = self.bullet_speed_y,
                                   velocity3 = self.bullet_speed_z )
        
        
        #***************************************************
        # INTERACTION: SURFACE-TO-SURFACE CONTACT
        #***************************************************
        interaction_properties = model.ContactProperty('IntProp-1')
        interaction_properties.TangentialBehavior( formulation        = abaqusConstants.PENALTY,
                                                   table              = ((0.5,),),
                                                   maximumElasticSlip = abaqusConstants.FRACTION,
                                                   fraction           = 0.005 )
        
        
        interaction_properties.NormalBehavior(pressureOverclosure=abaqusConstants.HARD)
        
        # definizione del contatto tra proiettile e plate
        model.SurfaceToSurfaceContactExp( name                = 'Int-1',
                                          createStepName      = 'Initial',
                                          main                = model.rootAssembly.instances['bullet'].surfaces['surface-all'],      # <--- aggiornato
                                          secondary           = model.rootAssembly.instances['plate'].surfaces['surface-top'],
                                          sliding             = abaqusConstants.FINITE,
                                          interactionProperty = 'IntProp-1' )
        
        
        #********************************************
        # CREATING PLATE MESH
        #********************************************
        # LIBRERIA ELEMENTI 3D:
        # https://classes.engineering.wustl.edu/2009/spring/mase5513/abaqus/docs/v6.6/books/usb/default.htm?startat=pt06ch22s01ael03.html#usb-elm-e3delem
        
        edge_top_north    = part_plate.edges.findAt( coordinates = (0,                   0,                  0) )
        edge_top_south    = part_plate.edges.findAt( coordinates = (0,                   0, self.plate_width  ) )
        edge_top_east     = part_plate.edges.findAt( coordinates = (+self.plate_width/2, 0, self.plate_width/2) )
        edge_top_west     = part_plate.edges.findAt( coordinates = (-self.plate_width/2, 0, self.plate_width/2) )
        
        edge_bottom_north = part_plate.edges.findAt( coordinates = (                  0, -self.plate_height,                  0) )
        edge_bottom_south = part_plate.edges.findAt( coordinates = (                  0, -self.plate_height, self.plate_width  ) )
        edge_bottom_east  = part_plate.edges.findAt( coordinates = (+self.plate_width/2, -self.plate_height, self.plate_width/2) )
        edge_bottom_west  = part_plate.edges.findAt( coordinates = (-self.plate_width/2, -self.plate_height, self.plate_width/2) )
        
        edge_ne           = part_plate.edges.findAt( coordinates = (+self.plate_width/2, -self.plate_height/2,                0) )
        edge_nw           = part_plate.edges.findAt( coordinates = (-self.plate_width/2, -self.plate_height/2,                0) )
        edge_se           = part_plate.edges.findAt( coordinates = (+self.plate_width/2, -self.plate_height/2, self.plate_width) )
        edge_sw           = part_plate.edges.findAt( coordinates = (-self.plate_width/2, -self.plate_height/2, self.plate_width) )
        
        
        # seed sugli edge orizzontali a double bias (cioe' un gradiente con tre valori)
        part_plate.seedEdgeByBias( biasMethod  = abaqusConstants.DOUBLE,
                                   centerEdges = ( edge_top_north, 
                                                   edge_top_south, 
                                                   edge_top_east, 
                                                   edge_top_west,
                                                   edge_bottom_north, 
                                                   edge_bottom_south, 
                                                   edge_bottom_east, 
                                                   edge_bottom_west ),
                                   minSize     = self.plate_seed_top_bottom_min, 
                                   maxSize     = self.plate_seed_top_bottom_max )
        
        
        # seed sugli edge verticali a single bias (cioe' un gradiente con due valori)
        part_plate.seedEdgeByBias( biasMethod = abaqusConstants.SINGLE, 
                                   end2Edges  = (edge_ne, edge_sw), 
                                   end1Edges  = (edge_nw, edge_se), 
                                   minSize    = self.plate_seed_sides_min, 
                                   maxSize    = self.plate_seed_sides_max )
        
        part_plate.generateMesh()
        
        
        #********************************************
        # MESH BULLET
        #********************************************
        part_bullet.seedPart( size = self.bullet_seed_size )
        
        # questa parte si puo' meshare solo se gli elementi hanno una forma tetraedrica, e gli va detto esplicitamente
        part_bullet.setMeshControls( regions   = part_bullet.cells, 
                                     elemShape = abaqusConstants.TET )
        
        part_bullet.generateMesh()
        
        
        if SAVEINPUTDATA:
        
            self._saveInputDataToFile()
        
        
        #***************
        # DEFINING A JOB
        #***************
        JOB_NAME = "Simulation_Job_" + str(self.index)
        job      = mdb.Job( name  = JOB_NAME, 
                            model = MODEL_NAME )
        
        
        #****************************************
        # SAVING INPUT FILE AS ("<NOME JOB>.INP")
        #****************************************
        if SAVEJOBINPUT:
            
            job.writeInput()
            
            
        #*******************
        # SUBMITTING THE JOB
        #*******************
        job.submit()
        job.waitForCompletion()
        
        lck_path = JOB_NAME + '.lck'
        # IF THIS FILE EXISTS, THE SIMULATION IS STILL RUNNING.
        while os.path.exists(lck_path):
            sleep(0.2)
        
        # CHECK IF SIMULATION HAS COMPLETED, I.E. IF THE BULLET HAS STOPPED
        bullet_region = session.openOdb(JOB_NAME + '.odb').steps['Step-1'].historyRegions['Node ASSEMBLY.1']
        bullet_v2Data = bullet_region.historyOutputs['V2_FILTER-1'].data
        
        simulation_length, bullet_final_velocity = bullet_v2Data[-1]
        
        # VARIABLE THAT SAYS WHETHER SIMULATION HAS COMPLETED
        simulation_completed = (bullet_final_velocity >= 0.0)
        
        
        #**************************
        # SAVING OUTPUT IN FILE CSV
        #**************************
        
        #*****************
        # LOADING DATABASE
        #*****************
        odb = session.openOdb(JOB_NAME + '.odb')
        
        total_time_lst     = np.array( object = [ frame.frameValue for frame in odb.steps['Step-1'].frames ], dtype = float )
        frames_to_identify = np.array( object = [2/30, 1/30], dtype = float )
        
        diff    = np.abs( total_time_lst[:, None] - frames_to_identify[None, :] )
        indices = np.argmin( a = diff, axis = 0 )
        # indices = np.sort( indices )
        
        print( f"Frames indices..: {indices}" ) #-->TO DELETE
        
        
        #************************
        # GETTING THE FIRST FRAME
        #************************
        firstFrame = odb.steps['Step-1'].frames[0]
        
        
        #**********************************************
        # GETTING THE CLOSEST FRAME AT 2/30 FROM IMPACT
        #**********************************************
        Frame_2_30 = None
        if indices[0]:
            
            Frame_2_30 = odb.steps['Step-1'].frames[indices[0]]
        
        
        #**********************************************
        # GETTING THE CLOSEST FRAME AT 1/30 FROM IMPACT
        #**********************************************
        Frame_1_30 = None
        if indices[1]:
            
            Frame_1_30 = odb.steps['Step-1'].frames[indices[1]]
        
        
        #************************
        # GETTING THE LAST FRAME
        #************************
        lastFrame  = odb.steps['Step-1'].frames[-1]
        
        
        #********************
        # REGIONS OF INTEREST
        #********************
        outputRegionExternal       = odb.rootAssembly.instances['PLATE'].nodeSets['SURFACE-ALL']
        outputRegionBulletExternal = odb.rootAssembly.instances['BULLET'].nodeSets['SURFACE-ALL']
        
        
        #************************
        # SAVING INCREMENT VALUES
        #************************
        increment_value_filename = os.path.join(self.new_path, 'increment_values.csv')
        
        increment_value_df = pd.DataFrame( { 'Increment Value' : total_time_lst } )

        increment_value_df.to_csv( path_or_buf = increment_value_filename, 
                                   index       = False)

        
        
        #********************
        # SAVING BULLET SPEED
        #********************
        if SAVEBULLETSPEED:
            
            region = odb.steps['Step-1'].historyRegions['Node ASSEMBLY.1']
            v2Data = region.historyOutputs['V2_FILTER-1'].data
            
            velocity_df = pd.DataFrame( { 'Time'     : [ time for time, _ in v2Data ] ,
                                          'Velocity' : [ v2   for _, v2   in v2Data ] } )
            
            velocity_output_filename = os.path.join( self.new_path, str(self.index) + '_bullet_velocity_y.csv' )
        
            velocity_df.to_csv( path_or_buf = velocity_output_filename, 
                                index       = False)
                                
                                
        #*************************
        # SAVING PLATE COORDINATES 
        #*************************
        if SAVEPLATECOORDINATES:
        
            #********************
            # INITIAL COORDINATES
            #********************
            coordinates_plate    = firstFrame.fieldOutputs['COORD'].getSubset(region = outputRegionExternal)
            coordinates_plate_df = pd.DataFrame( { 'Id'      : [ values.nodeLabel for values in coordinates_plate.values ],
                                                   'X_Coord' : [ values.data[0]   for values in coordinates_plate.values ],
                                                   'Y_Coord' : [ values.data[1]   for values in coordinates_plate.values ],
                                                   'Z_Coord' : [ values.data[2]   for values in coordinates_plate.values ] } )
                                                   
            coordinate_plate_filename = os.path.join(self.new_path, 'plate_initial_coordinates.csv')
            
            coordinates_plate_df.to_csv( path_or_buf = coordinate_plate_filename, 
                                         index       = False)
        
        
        #**************************
        # SAVING BULLET COORDINATES
        #**************************
        if SAVECOORDINATES:
        
            #*********************************************************
            #  SAVING BULLET COORDINATES AT 2/30 SECONDS BEFORE IMPACT
            #*********************************************************
            if Frame_2_30 is not None:
                
                coordinates_bullet_1    = Frame_2_30.fieldOutputs['COORD'].getSubset( region = outputRegionBulletExternal )
                coordinates_bullet_1_df = pd.DataFrame( { 'Id'      : [ values.nodeLabel for values in coordinates_bullet_1.values ],
                                                          'X_Coord' : [ values.data[0]   for values in coordinates_bullet_1.values ],
                                                          'Y_Coord' : [ values.data[1]   for values in coordinates_bullet_1.values ],
                                                          'Z_Coord' : [ values.data[2]   for values in coordinates_bullet_1.values ] } )
                                                          
                coordinate_bullet_1_filename = os.path.join(self.new_path, str(self.index) + '_input_coordinates_bullet_1.csv')
                
                coordinates_bullet_1_df.to_csv( path_or_buf = coordinate_bullet_1_filename, 
                                                index       = False)
        
        
            #*********************************************************
            # SAVING BULLET COORDINATES AT 1/30 SECONDS BEFORE IMPACT
            #*********************************************************
            if Frame_1_30 is not None:
                
                coordinates_bullet_2    = Frame_1_30.fieldOutputs['COORD'].getSubset( region = outputRegionBulletExternal )
                coordinates_bullet_2_df = pd.DataFrame( { 'Id'      : [ values.nodeLabel for values in coordinates_bullet_2.values ],
                                                          'X_Coord' : [ values.data[0]   for values in coordinates_bullet_2.values ],
                                                          'Y_Coord' : [ values.data[1]   for values in coordinates_bullet_2.values ],
                                                          'Z_Coord' : [ values.data[2]   for values in coordinates_bullet_2.values ] } )
                                                        
                coordinate_bullet_2_filename = os.path.join( self.new_path, str(self.index) + '_input_coordinates_bullet_2.csv' )
                
                coordinates_bullet_2_df.to_csv( path_or_buf = coordinate_bullet_2_filename, 
                                                index       = False )
        
        
        #***************************
        # SAVING PLATE DISPLACEMENTS
        #***************************
        if SAVEDISPLACEMENT:
        
            # ONLY THE EDGE OF THE PLATE
            displacement_external = lastFrame.fieldOutputs['U'].getSubset( region = outputRegionExternal )
        
            displacement_external_df = pd.DataFrame( { 'Id'     : [ values.nodeLabel for values in displacement_external.values ],
                                                       'X_Disp' : [ values.data[0]   for values in displacement_external.values ],
                                                       'Y_Disp' : [ values.data[1]   for values in displacement_external.values ],
                                                       'Z_Disp' : [ values.data[2]   for values in displacement_external.values ] } )
            
            displacement_external_output_filename = os.path.join( self.new_path, str(self.index) + '_output_displacement_external.csv' )
        
            displacement_external_df.to_csv( path_or_buf = displacement_external_output_filename, 
                                             index       = False )
        
        
        #************************
        # SAVING CURRENT DATABASE
        #************************
        if SAVEDATABASE:
            
            mdb.saveAs(str(self.index) + '.cae')
        
        
        #**************************************
        # ELIMINA FILE EXTRA GENERATI DA ABAQUS
        #**************************************
        files_ext = [ '.jnl',   '.sel', '.res', 
                      '.lck',   '.dat', '.msg', 
                      '.sta',   '.fil', '.sim',
                      '.stt',   '.mdl', '.prt', 
                      '.ipm',   '.log', '.com', 
                      '.odb_f', '.abq', '.pac',
                      '.rpy' ]
        
        
        if not SAVEJOBINPUT:
            
            files_ext.append('.inp')
        
        
        for file_ex in files_ext:
            
            file_path = JOB_NAME + file_ex
        
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass    
        if os.path.exists("abq.app_cache"):
            try:
                os.remove("abq.app_cache")
            except:
                pass    
        
        #******************************
        # RETURNING TO PARENT DIRECTORY
        #******************************
        os.chdir( self.previous_path )
        
        
        #**************
        # RELEASING ODB 
        #**************
        odb.close()
        
        
        return simulation_length, simulation_completed, indices

