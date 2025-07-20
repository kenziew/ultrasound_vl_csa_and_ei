function [settings_table, notes] = detect_resolution(dicom_struct)
      
    % Grab Pixel Information
    try
        resx = dicom_struct.SequenceOfUltrasoundRegions.Item_1.PhysicalDeltaX;
        resy = dicom_struct.SequenceOfUltrasoundRegions.Item_1.PhysicalDeltaY;
        unitx = dicom_struct.SequenceOfUltrasoundRegions.Item_1.PhysicalUnitsXDirection;
        unity = dicom_struct.SequenceOfUltrasoundRegions.Item_1.PhysicalUnitsYDirection;
    
        if unitx == 3 && unity == 3
            unit = 'cm';
        else
            unit = {};
        end
    catch
        resx = {};
        resy = {};
        unit = {};
    end
    
    % Grab study time Information
    try
        dc_date = info.StudyDate;
        aq_time = info.AcquisitionTime;
    catch
        dc_date = {};
        aq_time = {};
    end
   
    varNames = { 'date', 'time',  'resx', 'resy', 'units'};
    settings_table = table( {dc_date}, {aq_time}, {resx}, {resy}, {unit}, 'VariableNames', varNames); 
end

