from cafeh.cafeh_genotype import CAFEHGenotype, fit_cafeh_genotype
from cafeh.cafeh_summary import CAFEHSummary, fit_cafeh_summary, fit_cafeh_z
from cafeh.fitting import weight_ard_active_fit_procedure
from cafeh.model_queries import *
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finemap and colocalize multiple studies with CAFEH')

    parser.add_argument('--mode', default='genotype',
        help='which version of CAFEH to run options: \"genotype\", \"beta\", or \"z\"')
    parser.add_argument('--genotype', '-X',
        help='path to tab delimited genotype data, first column must contain SNP names, first row must contain sample names')
    parser.add_argument('--phenotype', '-Y',
        help='path to tab delimited phenotype data, first column must contain phenotype names, first row must contain sample names')
    parser.add_argument('--covariates', '-c',
        help='path to tab delimited covariate data, first column indicates phenotype, second column contains covariate name, first row contains sample names')
    parser.add_argument('--ld', '-R',
        help='path to tab delimited linkage disequilibrium matrix. First row and column contain SNP names')
    parser.add_argument('--zscores', '-z',
        help='path to tab delimited z score matrix. First column contains phenotype names, first row contains SNP names')
    parser.add_argument('--betas', '-B',
        help='path to tab delimited z score matrix. First column contains phenotype names, first row contains SNP names')
    parser.add_argument('--standard-errors', '-S',
        help='path to tab delimited z score matrix. First column contains phenotype names, first row contains SNP names')
    parser.add_argument('--sample-sizes', '-n',
        help='path to tab delimited sample size matrix. First column contains phenotype names, first row contains SNP names, entries are number of samples for each association test')
    parser.add_argument('--verbose',
        help='Print ELBO during optimization', action='store_true')
    parser.add_argument('--components', '-K', default=10, type=int,
        help='number of causal components (K) to fit CAFEH model with')

    # fit arguments
    parser.add_argument('--max-iter', default=100, type=int,
        help='maximum number of iterations')
    parser.add_argument('--no-ard', action='store_false',
        help='fit CAFEH without estimating prior variance of effect size')

    # initialization arguments
    parser.add_argument('--p0k', default=0.1, type=float,
        help='prior probability that each causal component is active in each phenotype')
    parser.add_argument('--prior-variance', default=0.1, type=float,
        help='initialize effect size variance, this choice should be made carefully if you are not estimating the effec size variances by using the \"--no-ard\" flag')
    parser.add_argument('--tolerance', default=1e-5, type=float,
        help='tolerance to declare convergnce, default = 1e-5')

    # output options
    parser.add_argument('--out', '-o', default='.',
        help='directory to save output to (creates directory if needed)')

    parser.add_argument('--save-model',
        help='save pickle of CAFEH object', action='store_true')



    # parse arguments
    args = parser.parse_args()

    # extract initialization arguments from args
    init_args = {
        'p0k': args.p0k,
        'prior_variance': args.prior_variance,
        'tolerance': args.tolerance
    }

    # extract initialization arguments from args
    fit_args = {
        'verbose': args.verbose,
        'max_iter': args.max_iter,
        'ARD_weights': args.no_ard
    }

    if args.mode == 'genotype':
        X_df = pd.read_csv(args.genotype, sep='\t', index_col=0)
        y_df = pd.read_csv(args.phenotype, sep='\t', index_col=0)
        cov_df = pd.read_csv(args.covariates, sep='\t', index_col=[0, 1])
        print('fitting CAFEH genotype...')
        cafeh = fit_cafeh_genotype(X_df, y_df, covariates=cov_df, K=args.components,
            init_args=init_args, fit_args=fit_args)
        
    if args.mode == 'beta':
        LD_df = pd.read_csv(args.ld, sep='\t', index_col=0)
        beta_df = pd.read_csv(args.betas, sep='\t', index_col=0)
        se_df = pd.read_csv(args.standard_errors, sep='\t', index_col=0)
        n_df = pd.read_csv(args.sample_sizes, sep='\t', index_col=0)

        print('fitting CAFEH with effect sizes and standard errors...')
        cafeh = fit_cafeh_summary(
            LD_df, beta_df, se_df, n_df, K=args.components,
            init_args=init_args, fit_args=fit_args)
        
    if args.mode == 'z':
        LD_df = pd.read_csv(args.ld, sep='\t', index_col=0)
        n_df = pd.read_csv(args.sample_sizes, sep='\t', index_col=0)

        # if z scores not provided, compute from betas and standard errors
        if args.zscores is None:
            print('computing z scores from summary statistics....')
            beta_df = pd.read_csv(args.betas, sep='\t', index_col=0)
            se_df = pd.read_csv(args.standard_errors, sep='\t', index_col=0)
            z_df = beta_df / se_df
        else:
            z_df = pd.read_csv(args.zscores, sep='\t', index_col=0)

        print('fitting CAFEH with z scores...')
        cafeh = fit_cafeh_z(
            LD_df, z_df, n_df, K=args.components,
            init_args=init_args, fit_args=fit_args)

    # make output directory if necessary
    os.makedirs(args.out, exist_ok=True)

    print('saving output...')

    # save model output
    variant_report = summary_table(cafeh)
    vr_path = args.out + '/' + 'cafeh.' + args.mode + '.results'
    print('saving results table to {}'.format(vr_path))
    variant_report.to_csv(vr_path, sep='\t')

    # save model
    if args.save_model:
        model_path = args.out + '/' + 'cafeh.' + args.mode + '.model'
        print('saving cafeh model to {}'.format(model_path))
        cafeh.save(model_path, save_data=False)


