import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Standardize;


public class InstanceUtils {
	public static Instances mergeInstances(Instances... set) throws Exception {
		Instances result = set[0];
		
		for (int i=1; i<set.length; i++) {
			// On enleve l'attribut de classe pour eviter les erreurs d'attribut
			// non unique.
			Remove remove = new Remove();
			remove.setOptions(new String[]{ "-R", set[i].numAttributes() + "" });
			remove.setInputFormat(set[i]);
			
			Instances instNew = Filter.useFilter(set[i], remove);
			
			// On merge avec l'ensemble que l'on a deja.
			result = Instances.mergeInstances(instNew, result);
		}
		
		result.setClassIndex(result.numAttributes() - 1);
		return result;
	}
	
	public static Instances removeIdentifier(Instances inst) throws Exception {
		Remove remove = new Remove();
		remove.setOptions(new String[]{ "-R", "1,2" });
		remove.setInputFormat(inst);
		
		Instances instNew = Filter.useFilter(inst, remove);
		instNew.setClassIndex(instNew.numAttributes() - 1);
		
		return instNew;
	}
	
	public static void standardize(DataModel buildModel, DataModel evalModel) throws Exception {
		Instances base_jmirmfccs = buildModel.jmirmfccs;
		Instances base_marsyas = buildModel.marsyas;
		Instances base_ssd = buildModel.ssd;
		
		buildModel.jmirmfccs = standardize(base_jmirmfccs, buildModel.jmirmfccs);
		buildModel.marsyas = standardize(base_marsyas, buildModel.marsyas);
		buildModel.ssd = standardize(base_ssd, buildModel.ssd);
		
		evalModel.jmirmfccs = standardize(base_jmirmfccs, evalModel.jmirmfccs);
		evalModel.marsyas = standardize(base_marsyas, evalModel.marsyas);
		evalModel.ssd = standardize(base_ssd, evalModel.ssd);
	}
	
	public static Instances standardize(Instances base, Instances unstandardize) throws Exception {
		Standardize standardize = new Standardize();
		standardize.setInputFormat(base);
		return Filter.useFilter(unstandardize, standardize);
	}

	public static Instances prefixAttributeName(String prefix, Instances rh) {
		Instances inst = new Instances(rh);
		
		for (int i=0; i<inst.numAttributes(); i++) {
			if (i == inst.classIndex()) continue;
			
			String newName = prefix + inst.attribute(i).name();
			inst.renameAttribute(i, newName);
		}
		
		return inst;
	}
}
