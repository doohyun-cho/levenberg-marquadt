function lm_plots ( t, y_dat, y_fit, sigma_y, cvg_hst, filename )
% lm_plots ( t, y_dat, y_fit, sigma_y, cvg_hst, filename )
% Plot statistics of the results of a Levenberg-Marquardt least squares
% analysis with lm.m

%   Henri Gavin, Dept. Civil & Environ. Engineering, Duke Univ. 2 May 2016

y_dat = y_dat(:);
y_fit = y_fit(:);

[max_it,n] = size(cvg_hst); n = n-3;

figure(101);  % plot convergence history of parameters, reduced chi^2, lambda
 clf
 subplot(211)
   plot( cvg_hst(:,1), cvg_hst(:,2:n+1), '-o','LineWidth',4);
   for i=1:n
     text(1.02*cvg_hst(max_it,1),cvg_hst(max_it,1+i), sprintf('%d',i) );
   end
   ylabel('parameter values')
 subplot(212)
  semilogy( cvg_hst(:,1) , [ cvg_hst(:,n+2) cvg_hst(:,n+3)], '-o','LineWidth',4)
   text(cvg_hst(1,1),cvg_hst(1,n+2), '\chi^2_\nu','FontSize',16,'color','k');
   text(cvg_hst(1,1),cvg_hst(1,n+3), '\lambda', 'FontSize',16, 'color','k');
   text(cvg_hst(max_it,1),cvg_hst(max_it,n+2), '\chi^2_\nu','FontSize',16,'color','k');
   text(cvg_hst(max_it,1),cvg_hst(max_it,n+3), '\lambda', 'FontSize',16, 'color','k');
   ylabel('\chi^2_\nu and \lambda')
   xlabel('function calls')
   print(sprintf('%sA.pdf', filename),'-dpdfcrop');

figure(102); % ------------ plot data, fit, and confidence interval of fit

 patchColor95 = [ 0.95, 0.95, 0.1 ];
 patchColor99 = [ 0.2,  0.95, 0.2 ];
 tp  = [ t   ; t(end:-1:1)   ; t(1)   ];            % x coordinates for patch
 yps95 =  y_fit + 1.96*sigma_y;                      %  + 95 CI
 yms95 =  y_fit - 1.96*sigma_y;                      %  - 95 CI
 yps99 =  y_fit + 2.58*sigma_y;                      %  + 99 CI
 yms99 =  y_fit - 2.58*sigma_y;                      %  - 99 CI
 yp95  = [ yps95 ; yms95(end:-1:1) ; yps95(1) ];    % y coordinates for patch
 yp99  = [ yps99 ; yms99(end:-1:1) ; yps99(1) ];    % y coordinates for patch

 clf
 hold on
   hc99 = patch(tp, yp99, 'FaceColor', patchColor99, 'EdgeColor', patchColor99);
   hc95 = patch(tp, yp95, 'FaceColor', patchColor95, 'EdgeColor', patchColor95);
   hd   = plot(t,y_dat,'ob'); 
   hf   = plot(t,y_fit,'-k');
 hold off
  axis('tight')
  legend([hd,hf,hc95,hc99], 'y_{data}','y_{fit}','95% c.i.','99% c.i.');
  ylabel('y(t)')
  xlabel('t')
  print(sprintf('%sB.pdf', filename),'-dpdfcrop');
 
figure(103); % ------------ plot histogram of residuals, are they Gaussean?
 clf
 hist(real(y_dat - y_fit))
  title('histogram of residuals')
  axis('tight')
  xlabel('y_{data} - y_{fit}')
  ylabel('count')
  print(sprintf('%sC.pdf', filename),'-dpdfcrop');
