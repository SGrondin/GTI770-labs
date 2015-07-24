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
	
	public static Instances standardize(Instances base, Instances unstandardize) throws Exception {
		Standardize standardize = new Standardize();
		standardize.setInputFormat(base);
		return Filter.useFilter(unstandardize, standardize);
	}
}
